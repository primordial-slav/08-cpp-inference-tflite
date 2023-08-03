#ifndef EXAMPLE_H  // An include guard to prevent double inclusion of this file
#define EXAMPLE_H

#include <string>  // Including another header file that this one depends on
#include "tensorflow/lite/interpreter.h"
#include <tensorflow/lite/c/c_api.h>
#include "tensorflow/lite/kernels/kernel_util.h"
#include <iostream>
#include <fstream>
#include <vector>


void filterClassesByScores(const float* raw_scores,
                           std::vector<float>& detection_scores,
                           std::vector<int>& detection_classes);

std::vector<std::pair<float, float>> ssd_generate_anchors();
void print_tensor_details(TfLiteTensor* tensor);
void writeVectorToFile(const std::vector<float>& data, const std::string& filename);
#endif  // Closing the include guard