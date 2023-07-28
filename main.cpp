#include <cstdio>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "decode_boxes.h"

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main() {
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile("../models/face_detection_full_range.tflite");

    TFLITE_MINIMAL_CHECK(model != nullptr);

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    interpreter->AllocateTensors();

    // Get the shape of the first input tensor
    TfLiteIntArray* input_dims = interpreter->input_tensor(0)->dims;
    printf("Input shape: ");
    for (int i = 0; i < input_dims->size; i++) {
        printf("%d ", input_dims->data[i]);
    }
    printf("\n");
    // Load the image
    cv::Mat img = cv::imread("../example.jpg", cv::IMREAD_COLOR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    if (img.empty()) {
        fprintf(stderr, "Failed to load image\n");
        exit(1);
    }
    // Get the input dimensions from the interpreter
    //int height = interpreter->input_tensor(0)->dims->data[1];
    //int width = interpreter->input_tensor(0)->dims->data[2];
    //int channels = interpreter->input_tensor(0)->dims->data[3];

    // Create a black image with the same dimensions as the model's input
    //cv::Mat img_dummy(height, width, channels == 1 ? CV_8UC1 : CV_8UC3, cv::Scalar(0));

    // Now you can use 'img' as the input to your model
    // Resize the image
    cv::resize(img, img, cv::Size(interpreter->input_tensor(0)->dims->data[1], interpreter->input_tensor(0)->dims->data[2]));
    //printf("Resized image shape: %d %d\n", img_dummy.rows, img_dummy.cols);
    float min_val = 0; // Set to your desired minimum value
    float max_val = 1; // Set to your desired maximum value

    cv::Mat tensor_data;
    img.convertTo(tensor_data, CV_32FC1, (max_val - min_val) / 255.0, min_val);
    cv::cvtColor(tensor_data, tensor_data, cv::COLOR_RGB2BGR);
    cv::imwrite("../resized_image.jpg", tensor_data);
    cv::cvtColor(tensor_data, tensor_data, cv::COLOR_BGR2RGB);
    // Copy the image data into the input tensor
    float* input = interpreter->typed_input_tensor<float>(0);
    memcpy(input, tensor_data.data, tensor_data.total() * tensor_data.elemSize());

    // Invoke the model
    interpreter->Invoke();

    // Get the output
    float* output = interpreter->typed_output_tensor<float>(0);
    // Open a file stream for output
    std::ofstream file("../output.txt");

    // First, run your model and get the output tensor.
    TfLiteTensor* output_tensor = interpreter->output_tensor(0);

    // Convert the output tensor to a cv::Mat.
    cv::Mat raw_boxes(output_tensor->dims->data[0], output_tensor->dims->data[1], CV_32F, output_tensor->data.f);

    // Then, call the function on the output.
    std::vector<cv::Mat> boxes = _decode_boxes(raw_boxes);

    // Now you can use 'boxes' however you need in your co
    // Check if any output values are NaN and write output to file
    int output_size = interpreter->output_tensor(0)->bytes / sizeof(float);
    for (int i = 0; i < output_size; i++) {
        if (std::isnan(output[i])) {
            printf("Output contains NaN values.\n");
            break;
        }
        file << output[i] << "\n";  // Write the output value to the file
    }

    // Don't forget to close the file when you're done
    file.close();

    // Check if any output values are NaN
    for (int i = 0; i < output_size; i++) {
        if (std::isnan(output[i])) {
            printf("Output contains NaN values.\n");
            break;
        }
    }
    // Get the shape of the first output tensor
    TfLiteIntArray* output_dims  = interpreter->output_tensor(0)->dims;
    printf("Output shape: ");
    for (int i = 0; i < output_dims ->size; i++) {
        printf("%d ", output_dims ->data[i]);
    }
    printf("\n");

    return 0;
}
