// decode_boxes.cpp
#include <vector>
#include <map>
#include <string>
#include <cassert>
#include <cmath>
#include "decode_boxes.h"
#include "anchors.h"
#include <iostream>
#include "tensorflow/lite/interpreter.h"
#include <tensorflow/lite/c/c_api.h>
#include "tensorflow/lite/kernels/kernel_util.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

#include <limits>
#include <unordered_set>

void filterClassesByScores(const float* raw_scores,
                           std::vector<float>& detection_scores,
                           std::vector<int>& detection_classes) {
    
    // Fixed 
    int num_classes = 1;
    int num_boxes = 2304; // TODO CHANGE THIS THING
    double sigmoid_score = 0.5;
    bool has_score_clipping_thresh = true;
    double score_clipping_thresh = 80;
    for (int i = 0; i < num_boxes; ++i) {
        int class_id = -1;
        float max_score = -std::numeric_limits<float>::max();
        // Find the top score for box i.
        for (int score_idx = 0; score_idx < num_classes; ++score_idx) {
            auto score = raw_scores[i * num_classes + score_idx];
            if (sigmoid_score) {
            if (has_score_clipping_thresh) {
                score = score < -score_clipping_thresh
                            ? -score_clipping_thresh
                            : score;
                score = score > score_clipping_thresh
                            ? score_clipping_thresh
                            : score;
            }
            score = 1.0f / (1.0f + std::exp(-score));
            }
            if (max_score < score) {
                max_score = score;
                class_id = score_idx;
            }
        }
        detection_scores[i] = max_score;
        detection_classes[i] = class_id;
    }
}


std::vector<std::pair<float, float>> ssd_generate_anchors() {
    int layer_id = 0;
    int num_layers = 1;
    std::vector<int> strides = {4};  // assuming strides is a vector in opts
    assert(strides.size() == num_layers);
    int input_height = 192;
    int input_width = 192;
    double anchor_offset_x = 0.5;
    double anchor_offset_y = 0.5;
    double interpolated_scale_aspect_ratio = 0.0;

    std::vector<std::pair<float, float>> anchors;

    while (layer_id < num_layers) {
        int last_same_stride_layer = layer_id;
        int repeats = 0;

        while (last_same_stride_layer < num_layers && strides[last_same_stride_layer] == strides[layer_id]) {
            last_same_stride_layer += 1;
            repeats += (interpolated_scale_aspect_ratio == 1.0) ? 2 : 1;
        }

        int stride = strides[layer_id];
        int feature_map_height = input_height / stride;
        int feature_map_width = input_width / stride;

        for (int y = 0; y < feature_map_height; y++) {
            double y_center = (y + anchor_offset_y) / feature_map_height;

            for (int x = 0; x < feature_map_width; x++) {
                double x_center = (x + anchor_offset_x) / feature_map_width;

                for (int i = 0; i < repeats; i++) {
                    anchors.push_back(std::make_pair(x_center, y_center));
                }
            }
        }

        layer_id = last_same_stride_layer;
    }

    return anchors;
}



void print_tensor_details(TfLiteTensor* tensor) {
    if (tensor == nullptr) {
        std::cout << "Tensor is nullptr." << std::endl;
        return;
    }
    // Print the output tensor type
    const char* type = TfLiteTypeGetName(tensor->type);

    std::cout << "Type: " << type << std::endl;

    // Print the number of dimensions
    std::cout << "Number of dimensions: " << tensor->dims->size << std::endl;

    // Print the shape of the tensor
    std::cout << "Shape: ";
        for (int i = 0; i < tensor->dims->size; ++i) {
            std::cout << tensor->dims->data[i] << " ";
    }
    std::cout << std::endl;
}


void writeVectorToFile(const std::vector<float>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (const float& value : data) {
            file << value << "\n"; // Write each value to a new line in the file
        }
        file.close();
        std::cout << "Data has been written to " << filename << std::endl;
    } else {
        std::cout << "Unable to open the file " << filename << std::endl;
    }
}

std::tuple<cv::Mat, int, int, int, int> resizeAndPad(cv::Mat& img, const cv::Size& targetSize, const cv::Scalar& padColor)
{
    cv::Mat res;

    // Calculate the aspect ratio of the image
    double aspectRatio = (double)img.cols / (double)img.rows;
    double targetAspectRatio = (double)targetSize.width / (double)targetSize.height;

    // Resize the image so that the dimension with the smaller ratio becomes 192
    if (aspectRatio > targetAspectRatio)
    {
        cv::resize(img, res, cv::Size(targetSize.width, std::round(targetSize.width / aspectRatio)), 0, 0, cv::INTER_AREA);
    }
    else
    {
        cv::resize(img, res, cv::Size(std::round(targetSize.height * aspectRatio), targetSize.height), 0, 0, cv::INTER_AREA);
    }

    // Calculate padding
    int topPad = (targetSize.height - res.rows) / 2;
    int bottomPad = targetSize.height - topPad - res.rows;
    int leftPad = (targetSize.width - res.cols) / 2;
    int rightPad = targetSize.width - leftPad - res.cols;

    // Pad the image
    cv::copyMakeBorder(res, res, topPad, bottomPad, leftPad, rightPad, cv::BORDER_CONSTANT, padColor);
    return std::make_tuple(res, topPad, bottomPad, leftPad, rightPad);
    //return res, topPad, bottomPad, leftPad, rightPad;
}

// Function to remove letterbox padding
void remove_letterbox(std::vector<std::vector<float>>& detections, 
                      int original_width, int original_height, 
                      int top_pad, int bottom_pad, int left_pad, int right_pad) {
    int inner_width = original_width - left_pad - right_pad;
    int inner_height = original_height - top_pad - bottom_pad;

    float scale_x = (float)original_width / (float)inner_width;
    float scale_y = (float)original_height / (float)inner_height;
    float offset_x = (float)left_pad;
    float offset_y = (float)top_pad;

    for (auto& detection : detections) {
        detection[0] = (detection[0] - offset_x) * scale_x; // xmin
        detection[1] = (detection[1] - offset_y) * scale_y; // ymin
        detection[2] = (detection[2] - offset_x) * scale_x; // xmax
        detection[3] = (detection[3] - offset_y) * scale_y; // ymax
    }
}