#include <cstdio>
#include <opencv2/opencv.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

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

  // Resize the image
  cv::resize(img, img, cv::Size(interpreter->input_tensor(0)->dims->data[1], interpreter->input_tensor(0)->dims->data[2]));
  printf("Resized image shape: %d %d\n", img.rows, img.cols);
  // Copy the image data into the input tensor
  float* input = interpreter->typed_input_tensor<float>(0);
  memcpy(input, img.data, img.total() * img.elemSize());

  // Invoke the model
  interpreter->Invoke();

  // Get the output
  float* output = interpreter->typed_output_tensor<float>(0);

  // Get the shape of the first output tensor
  TfLiteIntArray* output_dims  = interpreter->output_tensor(0)->dims;
  printf("Output shape: ");
  for (int i = 0; i < output_dims ->size; i++) {
    printf("%d ", output_dims ->data[i]);
  }
  printf("\n");

  return 0;
}
