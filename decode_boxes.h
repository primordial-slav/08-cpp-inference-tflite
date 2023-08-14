#pragma once
#ifndef EXAMPLE_H  // An include guard to prevent double inclusion of this file
#define EXAMPLE_H

#include <string>  // Including another header file that this one depends on
#include "tensorflow/lite/interpreter.h"
#include <tensorflow/lite/c/c_api.h>
#include "tensorflow/lite/kernels/kernel_util.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>


void filterClassesByScores(const float* raw_scores,
                           std::vector<float>& detection_scores,
                           std::vector<int>& detection_classes);

std::vector<std::pair<float, float>> ssd_generate_anchors();
void print_tensor_details(TfLiteTensor* tensor);
void writeVectorToFile(const std::vector<float>& data, const std::string& filename);
// Function declaration for letterbox removal
void remove_letterbox(std::vector<std::vector<float>>& detections, 
                      int original_width, int original_height, 
                      int top_pad, int bottom_pad, int left_pad, int right_pad);
std::tuple<cv::Mat, int, int, int, int> resizeAndPad(cv::Mat& img, const cv::Size& targetSize, const cv::Scalar& padColor);
#endif  // Closing the include guard