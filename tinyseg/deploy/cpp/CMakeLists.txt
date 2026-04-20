cmake_minimum_required(VERSION 3.10)
project(tinyseg_x5_segment_image_infer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  /usr/include
)

set(DNN_PATH /usr/lib)
set(HOBOT_PATH /usr/hobot/lib)
link_directories(
  /usr/lib
  ${HOBOT_PATH}
)

set(DNN_LIBS
  ${DNN_PATH}/libdnn.so
  ${DNN_PATH}/libhbrt_bayes_aarch64.so
  ${HOBOT_PATH}/libcnn_intf.so.1
  ${HOBOT_PATH}/libhbmem.so.1
  ${HOBOT_PATH}/libalog.so.1
)

add_executable(segment_image_infer
  segment_image_infer.cpp
  segment_infer.cpp
)

target_link_libraries(segment_image_infer
  ${OpenCV_LIBRARIES}
  ${DNN_LIBS}
)
