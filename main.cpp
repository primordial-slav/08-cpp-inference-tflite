#include <cstdio>
#include <fstream>
#include <iostream>
#include <chrono>  // for high_resolution_clock
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "decode_boxes.h"
#include "anchors.h"
#include "nms.hpp"

//#include "mediapipe.h"
//#include <mediapipe/framework/calculator_framework.pb.h>
#include <limits>
#include <unordered_set>

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }
    std::string image_path = argv[1];
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile("../models/face_detection_full_range.tflite");

    std::string output_folder = "../outputs/full_range/";

    // Extract the filename (substring that follows the last path separator)
    std::string filename;
    std::replace_copy(image_path.begin(), image_path.end(), std::back_inserter(filename), '/', '_');
    std::string draw_path = output_folder + filename;
    std::string raw_data_output = output_folder+filename+".txt";
    TFLITE_MINIMAL_CHECK(model != nullptr);

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);
    interpreter->SetNumThreads(1); //force to use cpu
    interpreter->AllocateTensors();

    // Get the shape of the first input tensor
    TfLiteIntArray* input_dims = interpreter->input_tensor(0)->dims;
    printf("Input shape: ");
    for (int i = 0; i < input_dims->size; i++) {
        printf("%d ", input_dims->data[i]);
    }
    printf("\n");

    // Load the image
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    auto start_time = std::chrono::high_resolution_clock::now();
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    if (img.empty()) {
        fprintf(stderr, "Failed to load image\n");
        exit(1);
    }
    // Resize the image
    //cv::resize(img, img, cv::Size(interpreter->input_tensor(0)->dims->data[1], interpreter->input_tensor(0)->dims->data[2]));
    int topPad;
    int bottomPad;
    int leftPad;
    int rightPad;
    cv::Mat res;
    std::tie(res, topPad, bottomPad, leftPad, rightPad)  = resizeAndPad(img, cv::Size(192, 192), cv::Scalar(0, 0, 0));
    
    
    //printf("Resized image shape: %d %d\n", img_dummy.rows, img_dummy.cols);
    float min_val = 0.; // Set to your desired minimum value
    float max_val = 1.; // Set to your desired maximum value

    cv::Mat tensor_data;
    res.convertTo(tensor_data, CV_32FC1); //, (max_val - min_val) / 255.0, min_val
    
    //cv::cvtColor(tensor_data, tensor_data, cv::COLOR_RGB2BGR);
    //cv::imwrite("../data/resized_image.jpg", tensor_data);
    //cv::cvtColor(tensor_data, tensor_data, cv::COLOR_BGR2RGB);
    tensor_data = tensor_data/255.0;

    
    // Copy the image data into the input tensor
    float* input = interpreter->typed_input_tensor<float>(0);
    memcpy(input, tensor_data.data, tensor_data.total() * tensor_data.elemSize());
    std::cout <<"Converted"<<std::endl;
    // Invoke the model
    interpreter->Invoke();
    
    // Get the output
    //float* output = interpreter->typed_output_tensor<float>(0);

    // First, run your model and get the output tensor.
    TfLiteTensor* raw_box_tensor = interpreter->output_tensor(0);
    TfLiteTensor* raw_score_tensor = interpreter->output_tensor(1);
    print_tensor_details(raw_box_tensor);
    std::vector<float> detection_scores(2304);
    std::vector<int> detection_classes(2304);

    // Now, write the data to a text file
    std::ofstream outfile(raw_data_output);
    if (!outfile) {
        std::cerr << "Failed to open the output file." << std::endl;
        return -1;
    }
    const float* raw_boxes = raw_box_tensor->data.f;
    const float* raw_scores = raw_score_tensor->data.f;
    std::vector<std::vector<float>> boxes;
    
    filterClassesByScores(raw_scores,detection_scores,detection_classes);
    for (int i = 0; i < 2304; ++i) {
        const int box_offset = i * 16; //+ options_.box_coord_offset()
        //int box_offset=0;
        float score = detection_scores[i];
        float x_center = raw_boxes[box_offset]/192.0 + anchorsArray[i][0]; // we know x is first because of https://github.com/patlevin/face-detection-tflite/blob/main/fdlite/types.py#L159
        float y_center = raw_boxes[box_offset + 1]/192.0 + anchorsArray[i][1];
        float w = raw_boxes[box_offset + 2]/192.0;
        float h = raw_boxes[box_offset + 3]/192.0;
        float half_size_w= w / 2.f;
        float half_size_h= h / 2.f;
        
        float xmin = (x_center - half_size_w)*192.0;
        float ymin = (y_center - half_size_h)*192.0;
        float xmax = (x_center + half_size_w)*192.0; //w
        float ymax = (y_center + half_size_h)*192.0; //h

        if (score > 0.01f ){ //&& score < 1.0f
            //outfile  <<"score: "<<score<< ", x: " << xmin << ", y: " << ymin << ", h: " << h << ", w: " << w <<std::endl;
            //int x1 = static_cast<int>(std::round(xmin));
            //int y1 = static_cast<int>(std::round(ymin));
            //int x2 = static_cast<int>(std::round(xmax));
            //int y2 = static_cast<int>(std::round(ymax));
            //outfile <<"score: "<<score<< ", x: " << x1 << ", y: " << y1 << ", x2: " << x2 << ", y2: " << y2 <<std::endl;
            //outfile <<"================================"<<std::endl;
            std::vector<float> box = {xmin,ymin,xmax,ymax,score,1};
            

            boxes.push_back(box);

        }
        
        
    }

    // NMS
    float iou_threshold = 0.3;
    nms(boxes,iou_threshold);
    // Letterbox removal
    remove_letterbox(boxes,192,192,topPad, bottomPad, leftPad, rightPad);
    auto end_time = std::chrono::high_resolution_clock::now();
    // Calculate the time difference between the starting and ending time points, in seconds
    auto duration = std::chrono::duration<double>(end_time - start_time);

    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    // Visualize
    // Iterate through the boxes
    cv::Mat draw = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::cvtColor(draw, draw, cv::COLOR_BGR2RGB);
    cv::resize(draw, draw, cv::Size(interpreter->input_tensor(0)->dims->data[1], interpreter->input_tensor(0)->dims->data[2]));
    for (const auto &box : boxes) {
        // Check if the inner vector has exactly 4 coordinates
        //if (box.size() != 4) continue;

        // Extract the coordinates
        float xmin = box[0];
        float ymin = box[1];
        float xmax = box[2];
        float ymax = box[3];
        float score = box[4];
        std::vector<float> save_box = {xmin,ymin,xmax,ymax,score,1};
        for (const auto& value : save_box) {
                outfile << value << ","; // Separate values with a space or any delimiter
            }
            outfile<<"\n";
            
        if (score > 0.5f ){
            // Create OpenCV Points for the top-left and bottom-right corners
            cv::Point2f topLeft(xmin, ymin);
            cv::Point2f bottomRight(xmax, ymax);

            // Draw a rectangle using the coordinates (color: red, thickness: 2)
            cv::rectangle(draw, topLeft, bottomRight, cv::Scalar(0, 255, 0), 2);
        }
    }
    cv::imwrite(draw_path, draw);
    return 0;
}
