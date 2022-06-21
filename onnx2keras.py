import warnings
from functools import partial

import onnx
from onnx import numpy_helper
import tensorflow as tf
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
import numpy as np
from tensorflow.python.ops.image_ops_impl import ResizeMethodV1


class Operations:
    def make_op(self, op_type, inputs, attrs):
        # print(op_type)
        # print([i.shape for i in inputs])
        # print(attrs)
        # print()
        return getattr(self, 'op_' + op_type.lower())(*inputs, **attrs)

class DataFormat: pass
class OnnxTensor(DataFormat): pass
class OnnxConstant(OnnxTensor): pass
class InterleavedImageBatch(DataFormat): pass

class OptimizationMissingWarning(Warning): pass

def ensure_data_format(tensor, format):
    if issubclass(tensor.data_format, format):
        return tensor
    elif tensor.data_format is OnnxConstant and format is InterleavedImageBatch:
        assert len(tensor.shape) == 4
        out = tensor.transpose([0, 2, 3, 1])
        out.data_format = InterleavedImageBatch
        return out
    elif tensor.data_format is OnnxTensor and format is InterleavedImageBatch:
        assert len(tensor.shape) == 4
        n, c, h, w = tensor.shape
        if h == w == 1 or c == 1:
            out = tf.reshape(tensor, [n, h, w, c])
        else:
            out = tf.transpose(tensor, [0, 2, 3, 1])
            warnings.warn("Transpose inserted. Please report at https://github.com/AxisCommunications/onnx-to-keras/issues", OptimizationMissingWarning)
        out.data_format = InterleavedImageBatch
        return out
    elif tensor.data_format is InterleavedImageBatch and format is OnnxTensor:
        assert len(tensor.shape) == 4
        n, h, w, c = tensor.shape
        if h == w == 1 or c == 1:
            out = tf.reshape(tensor, [n, c, h, w])
        else:
            out = tf.transpose(tensor, [0, 3, 1, 2])
            warnings.warn("Transpose inserted. Please report at https://github.com/AxisCommunications/onnx-to-keras/issues", OptimizationMissingWarning)
        out.data_format = OnnxTensor
        return out
    else:
        raise NotImplementedError

def compatible_data_format(format1, format2):
    return issubclass(format1, format2) or issubclass(format2, format1)

def ensure_compatible_data_format(a, b):
    if compatible_data_format(a.data_format, b.data_format):
        return a, b
    if b.data_format is OnnxConstant:
        if len(b.shape) == 0:
            return a, tf.broadcast_to(b, a.shape)
        else:
            return a, ensure_data_format(b, a.data_format)
    return ensure_data_format(a, b.data_format), b

class Constant(np.ndarray):
    data_format = OnnxConstant

class TfKerasOperations(Operations):
    keras = tf.keras
    make_tflite_compatible = False

    def parse_attr(self, a):
        if a.type == onnx.AttributeProto.INT:
            return a.i
        elif a.type == onnx.AttributeProto.INTS:
            return tuple(a.ints)
        elif a.type == onnx.AttributeProto.FLOAT:
            return a.f
        elif a.type == onnx.AttributeProto.STRING:
            return a.s
        elif a.type == onnx.AttributeProto.TENSOR:
            return self.make_constant(numpy_helper.to_array(a.t))
        else:
            raise NotImplementedError

    def make_constant(self, x):
        return np.asarray(x).view(Constant)

    def make_input(self, shape, dtype):
        dtype = tf.as_dtype(dtype)
        # XXX: Assumes all inputs are image batches that we want to transpose
        assert len(shape) == 4
        tensor = tf.keras.layers.Input((shape[2], shape[3], shape[1]), shape[0], None, dtype)
        tensor.data_format = InterleavedImageBatch
        return tensor

    def op_conv(self, x, weights, bias=None, kernel_shape=None, strides=None, pads=None, dilations=None, group=None):
        weights = ensure_data_format(weights, OnnxConstant)  # XXX Assumes no ops on weights
        x = ensure_data_format(x, InterleavedImageBatch)
        assert len(kernel_shape) == 2
        assert kernel_shape == weights.shape[2:4]

        conv_args = {
            "strides": strides,
            "dilation_rate": dilations,
            "kernel_size": kernel_shape,
        }

        if group > 1 and group == x.shape[3]: # Dephwise conv
            weights = weights.transpose(2, 3, 0, 1)
            ConvClass = self.keras.layers.DepthwiseConv2D
        elif not self.make_tflite_compatible or group == 1: # Regular conv
            weights = weights.transpose(2, 3, 1, 0)
            conv_args['filters'] = weights.shape[3]
            ConvClass = self.keras.layers.Conv2D
            conv_args['groups'] = group
        else: # Grouped conv
            # Grouped convolutions is supported in tf/keras but not yet supported in tflite
            # https://github.com/tensorflow/tensorflow/issues/40044
            warnings.warn(
                "Grouped conv splitted into {} regular convs for tflite compatibility.",
                OptimizationMissingWarning
            )
            class GroupedConv:
                def __init__(self, **kwargs):
                    self.groups, kwargs['groups'] = kwargs['groups'], 1
                    self.use_bias = kwargs['use_bias']
                    kwargs['filters'] //= self.groups
                    self.filters = kwargs['filters']

                    self.conv_layers = []
                    for _ in range(self.groups):
                        self.conv_layers.append(TfKerasOperations.keras.layers.Conv2D(**kwargs))

                def __call__(self, x):
                    splits = tf.split(x, self.groups, axis=-1)
                    convolved_splits = []
                    for split, layer in zip(splits, self.conv_layers):
                        convolved_splits.append(layer(split))

                    return tf.concat(convolved_splits, -1)

                def set_weights(self, w):
                    n = self.filters
                    for i, layer in enumerate(self.conv_layers):
                        grouped_w = w[0][:, :, :, i*n:(i+1)*n]
                        if self.use_bias:
                            grouped_b = w[1][i*n:(i+1)*n]
                            layer.set_weights([grouped_w.view(np.ndarray), grouped_b.view(np.ndarray)])
                        else:
                            layer.set_weights([grouped_w.view(np.ndarray)])

            weights = weights.transpose(2, 3, 1, 0)
            conv_args['filters'] = weights.shape[3]
            conv_args['groups'] = group
            ConvClass = GroupedConv

        conv_args['use_bias'] = not bias is None

        if pads == (0,0,0,0):
            conv_args['padding'] = 'valid'
        elif (kernel_shape[0] == kernel_shape[1] and pads[0] == pads[1] == pads[2] == pads[3] and
              pads[0] * 2 + 1 == kernel_shape[0] and strides == (1, 1) and dilations == (1, 1)):
            conv_args['padding'] = 'same'
        elif (kernel_shape == (3, 3) and pads == (1,1,1,1) and strides == (2,2) and dilations == (1, 1) and
              not (x.shape[1] == None or x.shape[2] == None) and x.shape[1] % 2 == 1 and x.shape[2] % 2 == 1):
            conv_args['padding'] = 'same'
        else:
            # ((top_pad, bottom_pad), (left_pad, right_pad))
            pad = self.keras.layers.ZeroPadding2D(((pads[0], pads[2]), (pads[1], pads[3])))
            x = pad(x)
            conv_args['padding'] = 'valid'

        conv = ConvClass(**conv_args)
        out = conv(x)
        out.data_format = InterleavedImageBatch

        if conv_args['use_bias']:
            conv.set_weights([weights.view(np.ndarray), bias.view(np.ndarray)])
        else:
            conv.set_weights([weights.view(np.ndarray)])

        return [out]

    def op_relu(self, x):
        out = self.keras.layers.ReLU()(x)
        out.data_format = x.data_format
        return [out]

    def op_leakyrelu(self, x, alpha):
        out = self.keras.layers.LeakyReLU(alpha=alpha)(x)
        out.data_format = x.data_format
        return [out]

    def op_sigmoid(self, x):
        out = self.keras.activations.sigmoid(x)
        out.data_format = x.data_format
        return [out]

    def op_softmax(self, x, axis):
        out = self.keras.activations.softmax(x, axis=axis)
        out.data_format = x.data_format
        return [out]

    def op_prelu(self, x, alpha):
        alpha = ensure_data_format(alpha, OnnxConstant)  # XXX Assumes no ops on alpha
        if len(alpha) == 1:
            shared = list(range(1, len(x.shape)))
            alpha = alpha.reshape((1,) * (len(x.shape) - 1))
        elif len(alpha) == x.shape[-1]:
            shared = list(range(1, len(x.shape) - 1))
        else:
            raise NotImplementedError
        alpha_initializer = self.keras.initializers.Constant(alpha.view(np.ndarray))
        out = self.keras.layers.PReLU(shared_axes=shared, alpha_initializer=alpha_initializer)(x)
        out.data_format = x.data_format
        return [out]

    def op_maxpool(self, x, kernel_shape, pads, strides, ceil_mode=0):
        assert ceil_mode == 0
        if len(kernel_shape) == 2:
            x = ensure_data_format(x, InterleavedImageBatch)
            if pads == (0, 0, 0, 0):
                padding = 'valid'
            else:
                # ((top_pad, bottom_pad), (left_pad, right_pad))
                pad = self.keras.layers.ZeroPadding2D(((pads[0], pads[2]), (pads[1], pads[3])))
                x = pad(x)
                padding = 'valid'
            out = self.keras.layers.MaxPool2D(kernel_shape, strides, padding)(x)
            out.data_format = InterleavedImageBatch
            return [out]
        else:
            raise NotImplementedError

    def op_concat(self, *tensors, axis):
        if all(t.data_format is InterleavedImageBatch for t in tensors):
            axis = (0, 3, 1, 2)[axis]
            out = self.keras.layers.Concatenate(axis)(list(tensors))
            out.data_format = InterleavedImageBatch
        elif all(t.data_format is OnnxConstant for t in tensors):
            out = self.make_constant(np.concatenate(tensors, axis))
        elif all(t.data_format is OnnxTensor for t in tensors):
            out = tf.concat(tensors, axis)
            out.data_format = OnnxTensor
        else:
            raise NotImplementedError
        return [out]

    def op_convtranspose(self, x, weights, bias=None, kernel_shape=None, strides=None, pads=None, dilations=None,
                         group=None, output_padding=(0, 0)):
        assert kernel_shape is not None
        assert strides is not None
        assert pads is not None
        assert dilations is not None
        assert group is not None
        weights = ensure_data_format(weights, OnnxConstant)  # XXX Assumes no ops on weights
        if bias is None:
            use_bias = False
            bias_initializer = None
        else:
            bias = ensure_data_format(bias, OnnxConstant)  # XXX Assumes no ops on weights
            use_bias = True

        if len(kernel_shape) == 2:
            x = ensure_data_format(x,  InterleavedImageBatch)
            assert kernel_shape == weights.shape[2:4]
            _, h_in, w_in, _ = x.shape
            h_out = (h_in - 1) * strides[0] - 2 * pads[0] + dilations[0] * (kernel_shape[0] - 1) + 1 + output_padding[0]
            w_out=(w_in - 1) * strides[1] - 2 * pads[1] + dilations[1] * (kernel_shape[1] - 1) + 1 + output_padding[1]


            if pads == (0,0,0,0):
                padding = 'valid'
            elif h_out == strides[0] * h_in and w_out == strides[1] * w_in and output_padding==(0,0):
                padding = 'same'
                output_padding = None  # output_padding overrides the padding argument in keras
            else:
                raise NotImplementedError
            # Tf; filter_height, filter_width, out_channels, in_channels
            # Torch: (in_channels, out_channels, kH, kW)
            weights = weights.transpose(2, 3, 1, 0)
            filters = weights.shape[2]
            if group == 1:
                conv = self.keras.layers.Conv2DTranspose(filters, kernel_shape, strides,
                                                         dilation_rate=dilations, padding=padding,
                                                         kernel_initializer='zeros',
                                                         use_bias=use_bias, bias_initializer='zeros',
                                                         output_padding=output_padding)
                out = conv(x)
                if use_bias:
                    conv.set_weights([weights.view(np.ndarray), bias.view(np.ndarray)])
                else:
                    conv.set_weights([weights.view(np.ndarray)])
            else:
                splits = tf.split(x, group, axis=-1)
                convolved_splits = []
                n = weights.shape[3] // group
                assert group * n == weights.shape[3]
                for i, split in enumerate(splits):
                    conv = self.keras.layers.Conv2DTranspose(filters, kernel_shape, strides,
                                                             dilation_rate=dilations, padding=padding,
                                                             kernel_initializer='zeros',
                                                             use_bias=use_bias, bias_initializer='zeros',
                                                             output_padding=output_padding)
                    convolved_splits.append(conv(split))
                    grouped_weights = weights[:, :, :, i*n:(i+1)*n]
                    if use_bias:
                        grouped_bias = bias[i*n:(i+1)*n]
                        conv.set_weights([grouped_weights.view(np.ndarray), grouped_bias.view(np.ndarray)])
                    else:
                        conv.set_weights([grouped_weights.view(np.ndarray)])
                out = tf.concat(convolved_splits, -1)

            assert out.shape[1] == h_out
            assert out.shape[2] == w_out
            out.data_format = InterleavedImageBatch
            return [out]
        else:
            raise NotImplementedError

    def op_batchnormalization(self, x, weight, bias, running_mean, running_var, momentum, epsilon):
        norm = self.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)
        out = norm(x)
        norm.set_weights([weight.view(np.ndarray), bias.view(np.ndarray),
                          running_mean.view(np.ndarray), running_var.view(np.ndarray)])
        out.data_format = x.data_format
        return [out]

    def op_unsqueeze(self, x, axes):
        x = ensure_data_format(x, OnnxTensor)
        out = x
        if isinstance(x, Constant):
            for ax in sorted(axes):
                out = np.expand_dims(out, ax).view(Constant)
            out.data_format = x.data_format
        else:
            for ax in sorted(axes):
                out = self.keras.backend.expand_dims(out, ax)
            out.data_format = OnnxTensor
        return [out]

    def op_clip(self, x, min, max):
        if min == 0:
            out = self.keras.layers.ReLU(max)(x)
        else:
            out = self.keras.backend.clip(x, min, max)
        out.data_format = x.data_format
        return [out]

    def op_add(self, x1, x2):
        x1, x2 = ensure_compatible_data_format(x1, x2)
        out = self.keras.layers.Add()([x1, x2])
        out.data_format = x1.data_format
        return [out]

    def op_sub(self, x1, x2):
        x1, x2 = ensure_compatible_data_format(x1, x2)
        out = self.keras.layers.Subtract()([x1, x2])
        out.data_format = x1.data_format
        return [out]

    def op_reducemean(self, x, axes, keepdims):
        x = ensure_data_format(x, InterleavedImageBatch)
        if axes == (2, 3) and keepdims == 0:
            out = self.keras.layers.GlobalAveragePooling2D()(x)
            out.data_format = OnnxTensor
        else:
            raise NotImplementedError

        return [out]

    def op_gemm(self, x, weights, bias, beta, transB, alpha):
        x = ensure_data_format(x, OnnxTensor)
        if beta == 1.0 and transB == 1 and alpha == 1.0:
            out = self.keras.layers.Dense(weights.shape[0], kernel_initializer='zeros',
                                          bias_initializer='zeros',
                                          weights=[weights.view(np.ndarray).T, bias.view(np.ndarray)])(x)
            out.data_format = OnnxTensor
        else:
            raise NotImplementedError
        return [out]

    def op_pad(self, x, pads, mode, value=0.0):
        x = ensure_data_format(x, InterleavedImageBatch)
        if mode == b'constant' and len(pads) == 8:
            assert len(x.shape) * 2 == len(pads)
            if pads[0] == pads[1] == pads[4] == pads[5] == 0:
                # ((top_pad, bottom_pad), (left_pad, right_pad))
                if value == 0.0:
                    paddings = ((pads[2], pads[6]), (pads[3], pads[7]))
                    out = self.keras.layers.ZeroPadding2D(paddings)(x)
                else:
                    paddings = ((0,0), (pads[2], pads[6]), (pads[3], pads[7]), (0,0))
                    out = tf.pad(x, paddings, constant_values=value)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        out.data_format = InterleavedImageBatch
        return [out]

    def op_averagepool(self, x, kernel_shape, pads, strides, ceil_mode=0):
        x = ensure_data_format(x, InterleavedImageBatch)
        assert ceil_mode == 0
        if len(x.shape) == 4:
            if pads == (0,0,0,0):
                padding = 'valid'
            else:
                raise NotImplementedError
            out = self.keras.layers.AveragePooling2D(kernel_shape, strides, padding)(x)
        else:
            raise NotImplementedError
        out.data_format = InterleavedImageBatch
        return [out]

    def op_globalaveragepool(self, x):
        x = ensure_data_format(x, InterleavedImageBatch)
        if len(x.shape) == 4:
            out = self.keras.backend.mean(x, axis=[1, 2], keepdims=True)
        else:
            raise NotImplementedError
        out.data_format = InterleavedImageBatch
        return [out]

    def op_flatten(self, x, axis):
        if axis == 1 and len(x.shape) == 4 and x.shape[1] == 1 and x.shape[2] == 1:
            out = self.keras.layers.Flatten()(x)
        else:
            raise NotImplementedError
        out.data_format = OnnxTensor
        return [out]

    def op_slice(self, x, starts, ends, axes=None, steps=None):
        if axes is None:
            axes = range(len(starts))
        if steps is None:
            steps = [1] * len(starts)
        if x.data_format is OnnxConstant:
            if axes != (0,):
                raise NotImplementedError
            out = self.make_constant(x[starts[0]:ends[0]:steps[0]])
        else:
            x = ensure_data_format(x, InterleavedImageBatch)
            if len(x.shape) != 4:
                raise NotImplementedError
            if len(axes) == 1 and starts[0] != ends[0]:
                if ends[0] == 2**63-1: # INT_MAX
                    if axes[0] == 0:
                        out = x[starts[0]::steps[0],:,:,:]
                    elif axes[0] == 1:
                        out = x[:,:,:,starts[0]::steps[0]]
                    elif axes[0] == 2:
                        out = x[:,starts[0]::steps[0],:,:]
                    elif axes[0] == 3:
                        out = x[:,:,starts[0]::steps[0],:]
                    else:
                        raise NotImplementedError
                else:
                    if axes[0] == 0:
                        out = x[starts[0]:ends[0]:steps[0],:,:,:]
                    elif axes[0] == 1:
                        out = x[:,:,:,starts[0]:ends[0]:steps[0]]
                    elif axes[0] == 2:
                        out = x[:,starts[0]:ends[0]:steps[0],:,:]
                    elif axes[0] == 3:
                        out = x[:,:,starts[0]:ends[0]:steps[0],:]
                    else:
                        raise NotImplementedError
            elif tuple(axes) == (2,3) and starts[0] != ends[0] and starts[1] != ends[1]:
                out = x[:,starts[0]:ends[0]:steps[0],starts[1]:ends[1]:steps[1],:]
            else:
                raise NotImplementedError
            out.data_format = InterleavedImageBatch
        return [out]

    def op_constant(self, value):
        out = value
        out.data_format = OnnxConstant
        return [out]

    def op_shape(self, x):
        shape = list(map(int, x.shape))
        if x.data_format is InterleavedImageBatch:
            n, h, w, f = shape
            shape = [n, f, h, w]
        return [self.make_constant(shape)]

    def op_gather(self, x, indices, axis=0):
        if x.data_format is OnnxConstant and axis == 0:
            return [self.make_constant(x[indices])]
        elif x.data_format is OnnxTensor:
            x = tf.gather(x, self.make_constant(indices), axis=axis)
            x.data_format = OnnxTensor
            return [x]
        else:
            raise NotImplementedError

    def op_cast(self, x, to):
        dtype = {
            0: None, # UNDEFINED
            1: np.float,
            2: np.uint8,
            3: np.int8,
            4: np.uint16,
            5: np.int16,
            6: np.int32,
            7: np.int64,
            8: str,
            9: np.bool,
            10: np.float16,
            11: np.double,
            12: np.uint32,
            13: np.uint64,
            14: np.complex64,
            15: np.complex128,
            # // Non-IEEE floating-point format based on IEEE754 single-precision
            # // floating-point number truncated to 16 bits.
            # // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
            #BFLOAT16 = 16;
        }[to]
        if x.data_format is OnnxConstant:
            return [self.make_constant(x.astype(dtype))]
        else:
            out = self.keras.backend.cast(x, dtype)
            out.data_format = x.data_format
            return [out]

    def op_mul(self, a, b):
        if b.shape == ():
            a, b = b, a
        if a.shape == ():
            out = a * b
            out.data_format = b.data_format
            return [out]
        a, b = ensure_compatible_data_format(a, b)
        if a.data_format is OnnxConstant:
            return [self.make_constant(a * b)]
        else:
            out = tf.keras.layers.Multiply()([a, b])
            out.data_format = a.data_format
            return [out]

    def op_floor(self, x):
        x = ensure_data_format(x, OnnxConstant)
        return [self.make_constant(np.floor(x))]

    def op_div(self, a, b):
        a = ensure_data_format(a, OnnxConstant)
        b = ensure_data_format(b, OnnxConstant)
        return [self.make_constant(a / b)]

    def op_upsample(self, x, scales, mode=b'nearest'):
        if mode == b'nearest':
            return self.op_resize(x, None, scales, coordinate_transformation_mode=b'asymmetric', nearest_mode=b'floor')
        if mode == b'linear':
            return self.op_resize(x, None, scales, coordinate_transformation_mode=b'align_corners', mode=b'linear', nearest_mode=b'floor')
        raise NotImplementedError

    def op_resize(self, x, roi, scales=None, sizes=None, *,
                  coordinate_transformation_mode=b"half_pixel", cubic_coeff_a=-0.75, exclude_outside=0,
                  extrapolation_value=0.0, mode=b"nearest", nearest_mode=b"round_prefer_floor"):
        assert cubic_coeff_a == -0.75
        assert exclude_outside == 0
        assert extrapolation_value == 0.0

        x = ensure_data_format(x, InterleavedImageBatch)

        if sizes is None:
            assert scales[0] == scales[1] == 1
            size = [int(x.shape[1] * scales[2]), int(x.shape[2] * scales[3])]
        else:
            assert sizes[0] == x.shape[0]
            assert sizes[1] == x.shape[3]
            size = sizes[2:4]

        if mode == b'nearest' and coordinate_transformation_mode == b'asymmetric' and nearest_mode==b'floor':
            out = tf.compat.v1.image.resize(x, size, ResizeMethodV1.NEAREST_NEIGHBOR)
        elif mode == b'linear' and coordinate_transformation_mode == b'align_corners':
            out = tf.compat.v1.image.resize(x, size, ResizeMethodV1.BILINEAR, align_corners=True)
        else:
            raise NotImplementedError
        out.data_format = InterleavedImageBatch
        return [out]

    def op_equal(self, x, y):
        x, y = ensure_compatible_data_format(x, y)
        out = self.keras.backend.equal(x, y)
        out.data_format = x.data_format
        return [out]

    def op_and(self, x, y):
        x, y = ensure_compatible_data_format(x, y)
        out = x & y
        out.data_format = x.data_format
        return [out]

    def op_greater(self, x, y):
        x, y = ensure_compatible_data_format(x, y)
        out = self.keras.backend.greater(x, y)
        out.data_format = x.data_format
        return [out]

    def op_reshape(self, x, shape):
        x = ensure_data_format(x, OnnxTensor)
        assert x.shape[0] == shape[0]
        out = self.keras.layers.Reshape(shape[1:])(x)
        out.data_format = OnnxTensor
        return [out]

    def op_transpose(self, x, perm):
        if x.data_format is InterleavedImageBatch and tuple(perm) == (0, 2, 3, 1):
            x = tf.identity(x)
            x.data_format = OnnxTensor
            return [x]
        x = ensure_data_format(x, OnnxConstant)
        x = tf.transpose(x, perm)
        x.data_format = OnnxConstant
        return [x]

    def op_matmul(self, x1, x2):
        x1 = ensure_data_format(x1, OnnxTensor)
        x2 = ensure_data_format(x2, OnnxTensor)
        if x1.data_format is OnnxConstant:
            x1 = tf.convert_to_tensor(x1)
        if x2.data_format is OnnxConstant:
            x2 = tf.convert_to_tensor(x2)
        if len(x1.shape) == 2:
            assert len(x2.shape) == 2
            out = self.keras.backend.dot(x1, x2)
        elif len(x1.shape) == 3:
            assert len(x2.shape) == 3
            assert x1.shape[0] == x2.shape[0] == 1
            out = self.keras.backend.dot(x1, x2)
            out = tf.reshape(out, (1, out.shape[1], out.shape[3]))
        elif len(x1.shape) == 4:
            assert len(x2.shape) == 4
            assert x1.shape[0] == x2.shape[0] == 1
            assert x1.shape[1] == x2.shape[1] == 1
            out = self.keras.backend.dot(x1, x2)
            out = tf.reshape(out, (1, 1, out.shape[2], out.shape[5]))
        else:
            raise NotImplementedError
        out.data_format = OnnxTensor
        return [out]

    def op_sqrt(self, x):
        out = self.keras.backend.sqrt(x)
        out.data_format = x.data_format
        return [out]

    def op_abs(self, x):
        out = self.keras.backend.abs(x)
        out.data_format = x.data_format
        return [out]

    def op_neg(self, x):
        out = -x
        out.data_format = x.data_format
        return [out]



def onnx2keras(onnx_model, make_tflite_compatible=False):
    tensors = {}
    ops = TfKerasOperations()
    ops.make_tflite_compatible = make_tflite_compatible

    for init in onnx_model.graph.initializer:
        tensors[init.name] = ops.make_constant(numpy_helper.to_array(init))

    model_inputs = []
    for input in onnx_model.graph.input:
        if input.name in tensors:
            continue
        shape = [d.dim_value if (d.dim_value > 0 and d.dim_param == "") else None
                 for d in input.type.tensor_type.shape.dim]
        dtype = TENSOR_TYPE_TO_NP_TYPE[input.type.tensor_type.elem_type]
        tensors[input.name] = ops.make_input(shape, dtype)
        model_inputs.append(tensors[input.name])

    for node in onnx_model.graph.node:
        inputs = [tensors[i] for i in node.input]
        attrs = {a.name: ops.parse_attr(a) for a in node.attribute}
        output_tensors = ops.make_op(node.op_type, inputs, attrs)
        assert len(output_tensors) == len(node.output)
        for n, t in zip(node.output, output_tensors):
            tensors[n] = t

    outputs = [tensors[o.name] for o in onnx_model.graph.output]
    return tf.keras.models.Model(model_inputs, outputs)


def verify(keras_model, onnx_model_file, decimals=2):
    import onnxruntime
    onnx_sess = onnxruntime.InferenceSession(onnx_model_file)

    onnx_inputs = onnx_sess.get_inputs()
    keras_inputs = keras_model.input
    if not isinstance(keras_inputs, list):
        keras_inputs = [keras_inputs]

    assert len(keras_inputs) == len(onnx_inputs)

    keras_indata = []
    onnx_indata = {}
    for onnx_input, keras_input in zip(onnx_inputs, keras_inputs):
        assert isinstance(onnx_input.shape[0], str) or keras_input.shape[0] == onnx_input.shape[0] # Batch
        assert isinstance(onnx_input.shape[1], str) or keras_input.shape[3] == onnx_input.shape[1] # Channels
        assert isinstance(onnx_input.shape[2], str) or keras_input.shape[1] == onnx_input.shape[2] # Height
        assert isinstance(onnx_input.shape[3], str) or keras_input.shape[2] == onnx_input.shape[3] # Width

        indata_shape = list(keras_input.shape)
        indata_shape[0] = 1 if not indata_shape[0] else indata_shape[0]
        indata_shape[1] = 224 if not indata_shape[1] else indata_shape[1]
        indata_shape[2] = 224 if not indata_shape[2] else indata_shape[2]
        indata_shape[3] = 3 if not indata_shape[3] else indata_shape[3]

        indata = np.random.rand(*indata_shape).astype(keras_input.dtype.as_numpy_dtype)
        keras_indata.append(indata)
        onnx_indata[onnx_input.name] = indata.transpose(0, 3, 1, 2)

    onnx_outdata = onnx_sess.run(None, onnx_indata)
    keras_outdata = keras_model.predict(keras_indata)

    if not isinstance(keras_outdata, list):
        keras_outdata = [keras_outdata]

    for onnx_out, keras_out in zip(onnx_outdata, keras_outdata):
        if len(keras_out.shape) == 4:
            warnings.warn("Found 4D output, assuming output is an image, transposing output when verifying model.")
            keras_out = keras_out.transpose(0, 3, 1, 2)
        try:
            np.testing.assert_almost_equal(onnx_out, keras_out, decimals)
            print("Output tensor matches to {} decimals!".format(decimals))
        except Exception as e:
            print(e)


def main(infile, outfile=None, export_saved_model=False, verify_model=True, make_tflite_compatible=False):
    if outfile is None:
        outfile = infile[:-5] if infile[-5:] == '.onnx' else infile
        outfile += '.h5'
    model = onnx2keras(onnx.load(infile), make_tflite_compatible)
    if export_saved_model:
        import tensorflow.compat.v1 as tf_v1
        tf_v1.keras.experimental.export_saved_model(model, export_saved_model)
    else:
        model.save(outfile)
    if verify_model:
        verify(model, infile)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
