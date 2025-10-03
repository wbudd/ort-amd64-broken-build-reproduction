#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <fcntl.h>

#define GUARD_ORT(_arg_) do { \
    OrtStatus * _status_ = (_arg_); \
    if (_status_) { \
        char const * msg{ort->GetErrorMessage(_status_)}; \
        std::cerr << msg << "\n"; \
        ort->ReleaseStatus(_status_); \
        return EXIT_FAILURE; \
    } \
} while (0)

auto MODEL_PATH{"/tmp/test/model.ort"};

////////////////////////////////////////////////////////////////////////////////
// test_inference() ////////////////////////////////////////////////////////////

static int test_inference() {
    // Perform ORT initialization.
    OrtApi const * ort    {OrtGetApiBase()->GetApi(ORT_API_VERSION)};
    OrtEnv       * ort_env{};
    GUARD_ORT(ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "", &ort_env));
    // Create an ORT session.
    OrtSession        * session        {};
    OrtSessionOptions * session_options{};
    GUARD_ORT(ort->CreateSessionOptions(&session_options));
    // Set just in case: ORT_ENABLE_ALL is the default for recent ORT versions.
    GUARD_ORT(
        ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL)
    );
    GUARD_ORT(OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0));
    GUARD_ORT(ort->CreateSession(
        ort_env, MODEL_PATH, session_options, &session
    ));

    { // Get this model's number of input and output layers.
        size_t input_layer_c {};
        size_t output_layer_c{};
        GUARD_ORT(ort->SessionGetInputCount (session, &input_layer_c ));
        GUARD_ORT(ort->SessionGetOutputCount(session, &output_layer_c));
        if (input_layer_c != 1 || output_layer_c != 1) {
            std::cerr << "This demo currently only supports ONNX/ORT models "
                "with a single input and output layer. Found " << input_layer_c
                << " input layer(s) and " << output_layer_c <<
                " output layer(s).\n";
            return EXIT_FAILURE;
        }
    }

    // Get this model's input layer name and the output layer name.
    OrtAllocator * allocator{}; // Does not need to be free()d.
    GUARD_ORT(ort->GetAllocatorWithDefaultOptions(&allocator));
    char * input_layer_name;
    char * output_layer_name;
    GUARD_ORT(ort->SessionGetInputName(
        session, 0, allocator, &input_layer_name
    ));
    GUARD_ORT(ort->SessionGetOutputName(
        session, 0, allocator, &output_layer_name
    ));

    // Get this model's input shape and output shape.
    OrtTypeInfo * type_info{};
    OrtTensorTypeAndShapeInfo const * tensor_info{};
    size_t dim_c{};
    GUARD_ORT(ort->SessionGetInputTypeInfo(session, 0, &type_info));
    GUARD_ORT(ort->CastTypeInfoToTensorInfo(type_info, &tensor_info));
    GUARD_ORT(ort->GetDimensionsCount(tensor_info, &dim_c));
    std::vector<int64_t> input_shape(dim_c);
    GUARD_ORT(ort->GetDimensions(tensor_info, &input_shape[0], dim_c));
    GUARD_ORT(ort->SessionGetOutputTypeInfo(session, 0, &type_info));
    GUARD_ORT(ort->CastTypeInfoToTensorInfo(type_info, &tensor_info));
    GUARD_ORT(ort->GetDimensionsCount(tensor_info, &dim_c));
    std::vector<int64_t> output_shape(dim_c);
    GUARD_ORT(ort->GetDimensions(tensor_info, &output_shape[0], dim_c));
    ort->ReleaseTypeInfo(type_info);

    // Print the obtained layer name and shape info, and calculate size totals.
    std::cout << "Loaded " << MODEL_PATH << " and obtained from it the "
        "following details: input_layer=\"" << input_layer_name <<
        "\", input_shape={";
    int64_t input_elem_c{1};
    for (size_t dim_i{}; dim_i < input_shape.size(); dim_i++) {
        if (dim_i) {
            std::cout << ", ";
        }
        auto dim_size{input_shape[dim_i]};
        std::cout    << dim_size;
        input_elem_c *= dim_size;
    }
    auto input_byte_c{static_cast<uint32_t>(input_elem_c * sizeof(float))};
    std::cout << "} (elem_c=" << input_elem_c << ", byte_c=" << input_byte_c <<
        "); output_layer=\"" << output_layer_name << "\", output_shape={";
    int64_t output_elem_c{1};
    for (size_t dim_i{}; dim_i < output_shape.size(); dim_i++) {
        if (dim_i) {
            std::cout << ", ";
        }
        auto dim_size{output_shape[dim_i]};
        std::cout     << dim_size;
        output_elem_c *= dim_size;
    }
    auto output_byte_c{static_cast<uint32_t>(output_elem_c * sizeof(float))};
    std::cout << "} (elem_c=" << output_elem_c << ", byte_c=" <<
        output_byte_c << ").\n";

    // These seem to be needed to prepare the input data for the inference call.
    OrtMemoryInfo * memory_info{};
    GUARD_ORT(ort->CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &memory_info
    ));
    OrtValue * ort_input {};
    OrtValue * ort_output{};

    std::vector<float> input_data(input_elem_c);

    // Load the dummy/empty input data as an ORT input tensor.
    // (In a real application, we'd fill input_data with actual data first).
    GUARD_ORT(ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        &input_data[0],
        input_byte_c,
        &input_shape[0],
        input_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &ort_input
    ));

    auto inference_start_time{std::chrono::system_clock::now()};

    // Perform the actual inference.
    GUARD_ORT(ort->Run(
        session, nullptr,
        &input_layer_name, &ort_input, 1,
        &output_layer_name, 1, &ort_output
    ));

    auto inference_end_time{std::chrono::system_clock::now()};
    auto inference_duration_ms{
        std::chrono::duration_cast<std::chrono::milliseconds>(
            inference_end_time - inference_start_time
        ).count()
    };
    std::cout << "Inference took " << inference_duration_ms << "ms.\n";

    // Obtain the output data.
    float * output_data{};
    GUARD_ORT(ort->GetTensorMutableData(
        ort_output,
        reinterpret_cast<void * *>(&output_data)
    ));

    // In a real application, we'd obviously do something here with output_data.

    std::cout << "Received " << output_byte_c << " bytes of inference output. "
        "Cleaning up.\n";

    // Clean up the ORT session.
    ort->ReleaseValue(ort_input);
    ort->ReleaseMemoryInfo(memory_info);
    ort->ReleaseSession(session);
    ort->ReleaseSessionOptions(session_options);

    std::cout << "Inference test succeeded. Exiting.\n";
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// Main() //////////////////////////////////////////////////////////////////////

int main() {
    try {
        return test_inference();
    } catch (std::runtime_error const & err) {
        std::cerr << "std::runtime_error thrown: " << err.what() << "\n";
    } catch (std::exception const & err) {
        std::cerr << "std::exception thrown: " << err.what() << "\n";
    } catch (...) {
        std::cerr << "Unrecognized exception thrown!\n";
    }
    std::cout << "Inference test failed. Exiting.\n";
    return EXIT_FAILURE;
}