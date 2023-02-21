END-OF-LIFE NOTICE
==================
Axis Communications is no longer maintaining this project

# onnx2keras

This is a tool for converting onnx models (as exported by for example pytorch) into tensorflow keras 
models. It focuses on inference performance and what we call high-level-compatibility rather than 
completeness. That is, it will not be able to convert every onnx model, but the models it can convert 
it will convert in a nice way. With high-level-compatibility we mean that the converted models produced
are constructed using the high-level keras API and should be similar to how the model would have 
been implemented in keras if it was implemented by hand.

Usage
-----
```
    python3 onnx2keras.py <infile.onnx> [<outfile.h5>]
```

Run tests
-----
```
    make
```
