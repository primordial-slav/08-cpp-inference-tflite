// decode_boxes.h

#ifndef DECODE_BOXES_H
#define DECODE_BOXES_H

#include <vector>
#include <opencv2/opencv.hpp>

std::vector<cv::Mat> _decode_boxes(cv::Mat raw_boxes);

#endif  // DECODE_BOXES_H