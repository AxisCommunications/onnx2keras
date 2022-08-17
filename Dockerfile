FROM tensorflow/tensorflow:1.15.2-py3
RUN pip install torch==1.11 torchvision
RUN pip install pytest
RUN pip install onnx
RUN pip install onnxruntime
RUN pip install fire
RUN mkdir /code
WORKDIR /code
