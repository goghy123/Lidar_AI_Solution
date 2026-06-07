# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.8)
project(cuda_bevfusion_ros2)

set(CMAKE_BUILD_PYTHON $ENV{USE_Python})
set(CMAKE_BUILD_TYPE "Release")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------
# ROS2 依赖
# -----------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# -----------------------------
# 原 CUDA BEVFusion 依赖
# -----------------------------
find_package(CUDA REQUIRED)
find_package(Protobuf REQUIRED)

# 你的工程放在 ros2_ws/src/CUDA-BEVFuison-ROS2，libraries/dependencies 与它同级。
# 使用绝对路径可以避免 colcon 在 build 目录执行时相对路径漂移。

set(spconv_include libraries/3DSparseConvolution/libspconv/include)
set(spconv_lib libraries/3DSparseConvolution/libspconv/lib/${arch}_cuda$ENV{SPCONV_CUDA_VERSION})
message(STATUS "spconv_lib = ${spconv_lib}")

if("$ENV{SPCONV_CUDA_VERSION}" STREQUAL "11.4")
  # For cuda 11.x
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB -Wno-unused-parameter")
  set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")
else()
  # For cuda 12.x
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB -Wno-unused-parameter")
  set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  -std=c++17
  --expt-relaxed-constexpr
  -gencode arch=compute_$ENV{CUDASM},code=sm_$ENV{CUDASM}
  -gencode arch=compute_$ENV{CUDASM},code=compute_$ENV{CUDASM}
)

include_directories(
  ${Protobuf_INCLUDE_DIRS}
  $ENV{CUDA_Inc}
  $ENV{TensorRT_Inc}
  $ENV{Python_Inc}
  dependencies/stb
  dependencies/dlpack/include
  dependencies/pybind11/include
  libraries/cuOSD/src
  ${spconv_include}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/common
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

link_directories(
  $ENV{CUDA_Lib}
  $ENV{TensorRT_Lib}
  ${spconv_lib}
  ${CMAKE_CURRENT_BINARY_DIR}
  $ENV{Python_Lib}
)

file(GLOB_RECURSE CORE_FILES
  src/bevfusion/*.cu
  src/bevfusion/*.cpp
  src/onnx/*.cpp
  src/common/tensor.cu
  src/common/tensorrt.cpp
)

file(GLOB_RECURSE VISUALIZE_FILES
  libraries/cuOSD/src/*.cpp
  libraries/cuOSD/src/*.cu
  src/common/visualize.cu
)

set(BEVFUSION_APP_FILES
  src/bevfusion_app/dataset_io.cpp
  src/bevfusion_app/core_factory.cpp
  src/bevfusion_app/visualization_renderer.cpp
  src/bevfusion_app/bevfusion_runner.cpp
)

####################### libbevfusion_core.so ##########################
cuda_add_library(bevfusion_core SHARED
  ${CORE_FILES}
)

target_link_libraries(bevfusion_core
  libcudart.so
  libcublasLt.so
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
)
#######################################################################

####################### 原 bevfusion 命令行程序 ########################
# 保留 src/main.cpp 的原 demo 入口，方便不通过 ROS2 时继续做快速调试。
cuda_add_executable(bevfusion
  src/main.cpp
  ${VISUALIZE_FILES}
)

target_link_libraries(bevfusion
  bevfusion_core
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
)
#######################################################################

####################### ROS2 数据集播放节点 ############################
cuda_add_executable(bevfusion_dataset_node
  src/ros2/bevfusion_dataset_node.cpp
  ${BEVFUSION_APP_FILES}
  ${VISUALIZE_FILES}
)

target_link_libraries(bevfusion_dataset_node
  bevfusion_core
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
  dl
)

ament_target_dependencies(bevfusion_dataset_node
  rclcpp
  sensor_msgs
  std_msgs
)
#######################################################################

########################## custom_layernorm.so ########################
cuda_add_library(custom_layernorm SHARED
  src/plugins/custom_layernorm.cu
)

target_link_libraries(custom_layernorm
  libnvinfer.so
  libnvinfer_plugin.so
)
#######################################################################

########################## libpybev.so ################################
if("${CMAKE_BUILD_PYTHON}" STREQUAL "ON")
  cuda_add_library(pybev SHARED
    src/python.cpp
  )
  target_link_libraries(pybev
    bevfusion_core
    libnvinfer.so
    libspconv.so
    ${Protobuf_LIBRARIES}
    libnvinfer_plugin.so
    $ENV{Python_Soname}
  )
endif()
#######################################################################

# -----------------------------
# install：ros2 run 只能找到安装到 lib/${PROJECT_NAME} 下的可执行文件。
# shared library 安装到 lib，source install/setup.bash 后会进入 LD_LIBRARY_PATH。
# -----------------------------
install(TARGETS
  bevfusion
  bevfusion_dataset_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  bevfusion_core
  custom_layernorm
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

if("${CMAKE_BUILD_PYTHON}" STREQUAL "ON")
  install(TARGETS pybev
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
endif()

install(DIRECTORY
  configs
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
