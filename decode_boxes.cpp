// decode_boxes.cpp

#include "decode_boxes.h"

std::vector<cv::Mat> _decode_boxes(cv::Mat raw_boxes) {
    // Implement your function here.
    // This is a placeholder and will not work as expected.
    // Replace with your actual implementation.

    int scale = raw_boxes.cols;
    int num_points = raw_boxes.cols / 2;
    std::vector<cv::Mat> boxes;

    // Use OpenCV functions to manipulate raw_boxes and populate boxes.
    // You can use OpenCV functions such as reshape, divide, multiply, add, subtract, etc.
    // Refer to the OpenCV documentation for details on how to use these functions.

    return boxes;
}