#pragma once

#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "bevfusion/head-transbbox.hpp"
#include "common/tensor.hpp"

namespace bevfusion_app {

struct RgbImage {
  int width = 0;
  int height = 0;
  int channels = 3;
  std::vector<unsigned char> data;

  bool empty() const { return data.empty(); }
};

// 把原 main.cpp 的 visualize 拆成“渲染到内存”，这样 ROS2 可以直接发布 Image，避免先写 jpg 再读 jpg。
RgbImage render_visualization(const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes,
                              const nv::Tensor& lidar_points,
                              const std::vector<unsigned char*>& images,
                              const nv::Tensor& lidar2image,
                              cudaStream_t stream);

// 可选保存 jpg。ROS2 第一阶段主要发布图像，但保留 outputs 落盘能力方便和原项目结果对比。
bool save_rgb_image_as_jpeg(const std::string& save_path, const RgbImage& image, int quality = 100);

}  // namespace bevfusion_app
