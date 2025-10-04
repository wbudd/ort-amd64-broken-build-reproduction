# Supported values are "x86_64" and "aarch64".
ARG platform=x86_64

# This should be the filename of the original ONNX model sans ".onnx" extension.
ARG onnx_model_name=msg3d

################################################################################
### Set up base AWS Lambda image ###############################################

# Use the vanilla AWS Lambda image as our base image to ensure interoperability.
# Choose a base image flavor depending on the --build-arg platform variable.
FROM public.ecr.aws/lambda/python:3.12       AS aws-lambda-python-x86_64
FROM public.ecr.aws/lambda/python:3.12-arm64 AS aws-lambda-python-aarch64
FROM aws-lambda-python-${platform}           AS ort-build

ARG platform
ARG onnx_model_name

LABEL maintainer="Will Budd"

WORKDIR /

RUN dnf -y install \
        binutils \
        gcc-c++ \
        git \
        glibc-devel \
        make \
        nasm \
        patch \
        tar \
        wget \
        yasm \
        zlib-devel \
    && dnf -y clean all \
    && rm -rf /var/cache

################################################################################
### Build CMake ################################################################

# The CMake packaged via the Lambda image's dnf is too old: build from source.
# (ONNX Runtime requires CMake version 3.28 or higher, but dnf provides 3.22.2.)
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8-Linux-${platform}.sh \
    && bash cmake-3.31.8-Linux-${platform}.sh \
        --exclude-subdir \
        --prefix=/usr/local \
        --skip-license \
    && rm cmake-3.31.8-Linux-${platform}.sh

################################################################################
### Perform a custom size-optimized build of ONNX Runtime as a static library ##

RUN git clone \
        --depth 1 \
        --recursive \
        https://github.com/microsoft/onnxruntime \
    && pip install onnx onnxruntime

COPY ${onnx_model_name}.onnx /

# There seems to be a bug when generating an optimized ORT on machines equipped
# with an AMD CPU, in which case the following error occurs at inference time:
#
# > Non-zero status code returned while running Conv node. Name:'878_nchwc'
# > Status Message: Input channels C is not equal to kernel channels * group.
# > C: 24 kernel channels: 32 group: 1
#
# No problems on Intel though, so when running inference on an AMD machine,
# it seems best to Docker build on an Intel machine, considering that the
# result is the same X86-64 bytecode.
#
# (And aarch64 to aarch64 seems fine too. E.g., M1 to Graviton2)

RUN if [[ ${platform} == "aarch64" ]]; \
        then ort_platform=arm; \
        else ort_platform=amd64; \
    fi; \
    python /onnxruntime/tools/python/convert_onnx_models_to_ort.py \
        --enable_type_reduction \
        --optimization_style Fixed \
        --save_optimized_onnx_model \
        --target_platform ${ort_platform} \
        /${onnx_model_name}.onnx

RUN cd onnxruntime \
    && ./build.sh \
        --allow_running_as_root \
        --build_dir build \
        # MinSizeRel is surprisingly(?) slightly faster than --config=Release
        --config=MinSizeRel \
        --disable_exceptions \
        --disable_ml_ops \
        --disable_rtti \
        --enable_reduced_operator_type_support \
        --include_ops_by_config \
            /${onnx_model_name}.required_operators_and_types.config \
        --minimal_build \
        --skip_tests

################################################################################
### Build test_inference.cpp with static linkage ###############################

COPY test_inference.cpp /tmp/

RUN g++ \
    -o /tmp/test_inference \
    -std=c++20 -Wall -Wextra -Wfatal-errors \
    -O3 \
    /tmp/test_inference.cpp \
	-I/onnxruntime/include/onnxruntime/core/session \
	-I/onnxruntime/include/onnxruntime/core/providers/cpu \
	/onnxruntime/build/MinSizeRel/libonnxruntime_session.a \
	/onnxruntime/build/MinSizeRel/libonnxruntime_providers.a \
	/onnxruntime/build/MinSizeRel/libonnxruntime_framework.a \
    /onnxruntime/build/MinSizeRel/libonnxruntime_lora.a \
	/onnxruntime/build/MinSizeRel/libonnxruntime_optimizer.a \
	/onnxruntime/build/MinSizeRel/libonnxruntime_graph.a \
	/onnxruntime/build/MinSizeRel/libonnxruntime_common.a \
	/onnxruntime/build/MinSizeRel/libonnxruntime_util.a \
	/onnxruntime/build/MinSizeRel/libonnxruntime_flatbuffers.a \
	/onnxruntime/build/MinSizeRel/libonnxruntime_mlas.a \
	/onnxruntime/build/MinSizeRel/_deps/onnx-build/libonnx.a \
	/onnxruntime/build/MinSizeRel/_deps/onnx-build/libonnx_proto.a \
	/onnxruntime/build/MinSizeRel/_deps/protobuf-build/libprotobuf-lite.a \
	/onnxruntime/build/MinSizeRel/_deps/abseil_cpp-build/absl/container/libabsl_raw_hash_set.a \
    /onnxruntime/build/MinSizeRel/_deps/abseil_cpp-build/absl/container/libabsl_hashtablez_sampler.a \
	/onnxruntime/build/MinSizeRel/_deps/abseil_cpp-build/absl/hash/libabsl_hash.a \
	/onnxruntime/build/MinSizeRel/_deps/abseil_cpp-build/absl/hash/libabsl_city.a \
	/onnxruntime/build/MinSizeRel/_deps/abseil_cpp-build/absl/hash/libabsl_low_level_hash.a \
    /onnxruntime/build/MinSizeRel/_deps/abseil_cpp-build/absl/synchronization/libabsl_synchronization.a \
    /onnxruntime/build/MinSizeRel/_deps/abseil_cpp-build/absl/synchronization/libabsl_kernel_timeout_internal.a \
    /onnxruntime/build/MinSizeRel/_deps/abseil_cpp-build/absl/time/libabsl_time.a \
    /onnxruntime/build/MinSizeRel/_deps/abseil_cpp-build/absl/time/libabsl_time_zone.a \
    /onnxruntime/build/MinSizeRel/_deps/abseil_cpp-build/absl/base/libabsl_base.a \
	/onnxruntime/build/MinSizeRel/_deps/abseil_cpp-build/absl/base/libabsl_throw_delegate.a \
	/onnxruntime/build/MinSizeRel/_deps/abseil_cpp-build/absl/base/libabsl_raw_logging_internal.a \
    /onnxruntime/build/MinSizeRel/_deps/abseil_cpp-build/absl/base/libabsl_malloc_internal.a \
    /onnxruntime/build/MinSizeRel/_deps/abseil_cpp-build/absl/base/libabsl_spinlock_wait.a \
    /onnxruntime/build/MinSizeRel/_deps/abseil_cpp-build/absl/debugging/libabsl_stacktrace.a \
    /onnxruntime/build/MinSizeRel/_deps/abseil_cpp-build/absl/debugging/libabsl_debugging_internal.a \
	/onnxruntime/build/MinSizeRel/_deps/pytorch_cpuinfo-build/libcpuinfo.a \
	-lpthread \
	-ldl

################################################################################
### Test the binary in an empty AWS Lambda image, and generate zip files #######

FROM public.ecr.aws/lambda/python:3.12       AS aws-lambda-python-reset-x86_64
FROM public.ecr.aws/lambda/python:3.12-arm64 AS aws-lambda-python-reset-aarch64
FROM aws-lambda-python-reset-${platform} AS test-and-zip

ARG platform
ARG onnx_model_name

COPY --from=ort-build /tmp/test_inference /tmp/test/test_inference
COPY --from=ort-build /${onnx_model_name}.ort /tmp/test/model.ort

# Create the output ZIP even if test_inference fails (due to the "||").
RUN /tmp/test/test_inference \
    && echo "#################### INFERENCE TEST SUCCEEDED ##################" \
    || echo "#################### INFERENCE TEST FAILED #####################"

RUN dnf -y install zip \
    && cd /tmp \
    && mv test/model.ort test/${onnx_model_name}_${platform}.ort \
    && output_name=test_inference_${onnx_model_name}_${platform} \
    && mv test ${output_name} \
    && zip -mr ${output_name}.zip ${output_name} \
    # Print the file size of the resulting zip files.
    && ls -ahl ${output_name}.zip

################################################################################
### Output the resulting zip file ##############################################

FROM scratch AS zip-output

ARG platform
ARG onnx_model_name

COPY --from=test-and-zip \
    /tmp/test_inference_${onnx_model_name}_${platform}.zip /