# ORT amd64 broken build reproduction

This repository is intended as a self-contained reproduction of issue https://github.com/microsoft/onnxruntime/issues/26174

## Steps to reproduce

1. `git clone` this repo in an x86_64 (either AMD or Intel) environment with a new(-ish) version of Docker installed.
2. With this repository as the current working directory, execute the following Docker command:

```
DOCKER_BUILDKIT=1 docker build -t ort-amd64-broken-build-reproduction:x86_64 --build-arg onnx_model_name=msg3d --build-arg platform=x86_64 --force-rm --progress=plain --output . .
```

## Dockerfile contents

The included Dockerfile executed by the command above runs all the necessary steps to reproduce the issue:

* Compilation of ONNX Runtime as static libraries.
* Conversion/optimization of included ONNX model.
* Inclusion and compilation of inference test program.
* Execution of statically compiled test program in a separate "clean-room" image.
* Output of ZIP files of generated ORT model and program.

## Expected result

The Docker build command listed above is expected to complete without fail, and produce an output ZIP file in the current working directory (the root of this repo).

That said, log outputs are expected to differ depending on the type of CPU used:

### If compiled on an Intel CPU:

The inclusion of the following line in the Docker build log output:

`#################### INFERENCE TEST SUCCEEDED #####################`.

### If compiled on an AMD CPU:

The inclusion of the following lines in the Docker build log output:

```
[E:onnxruntime:, sequential_executor.cc:572 ExecuteKernel] Non-zero status code returned while running Conv node. Name:'/sgcn1/sgcn1.0/mlp/layers.2/Relu_output_0_nchwc' Status Message: Input channels C is not equal to kernel channels * group. C: 24 kernel channels: 32 group: 1
```
And `#################### INFERENCE TEST FAILED #####################`.

## Reduced optimization level workaround for AMD

Setting the Docker `--build-arg amd_reduced_optimization_workaround=true` sets the ORT optimization level to `extended` instead of the default of `all` prior to model conversion. When this level is set to `extended`, the above error/bug no longer appears.

```
DOCKER_BUILDKIT=1 docker build -t ort-amd64-broken-build-reproduction:x86_64 --build-arg onnx_model_name=msg3d --build-arg platform=x86_64 --build-arg amd_reduced_optimization_workaround=true --force-rm --progress=plain --output . .
```