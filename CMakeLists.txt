cmake_minimum_required(VERSION 3.16)
project(minimal C CXX)

set(TENSORFLOW_SOURCE_DIR "/host/Boris/99-bazel_build/08-build-tflite/tensorflow_src" CACHE PATH
  "Directory that contains the TensorFlow project" )
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV and TensorFlow headers
include_directories(${OpenCV_INCLUDE_DIRS} ../include)

# Link OpenCV and TensorFlow libraries
link_directories(${OpenCV_LIBS} ../lib)

add_executable(main main.cpp decode_boxes.cpp)
target_link_libraries(main ${OpenCV_LIBS} tensorflow-lite)