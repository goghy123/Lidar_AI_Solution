#include "bevfusion_app/bevfusion_runner.hpp"

#include <stdexcept>
#include <vector>

#include "bevfusion_app/dataset_io.hpp"
#include "bevfusion_app/visualization_renderer.hpp"
#include "common/check.hpp"
#include "common/tensor.hpp"

namespace bevfusion_app {

BevFusionRunner::BevFusionRunner(const RunnerConfig& config) : config_(config) {
  core_ = create_core_from_config(config_.core);
  if (core_ == nullptr) {
    throw std::runtime_error("BEVFusion core 创建失败，请检查模型路径和 TensorRT engine 文件");
  }

  checkRuntime(cudaStreamCreate(&stream_));

  core_->print();
  core_->set_timer(config_.enable_core_timer);
}

BevFusionRunner::~BevFusionRunner() {
  // CUDA stream 必须和 runner 生命周期绑定，避免节点退出时资源泄漏。
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

FrameResult BevFusionRunner::process_frame(const std::string& frame_dir) {
  FrameResult result;
  result.frame_dir = frame_dir;
  result.frame_name = basename_of_path(frame_dir);

  auto camera2lidar = nv::Tensor::load(nv::format("%s/camera2lidar.tensor", frame_dir.c_str()), false);
  auto camera_intrinsics = nv::Tensor::load(nv::format("%s/camera_intrinsics.tensor", frame_dir.c_str()), false);
  auto lidar2image = nv::Tensor::load(nv::format("%s/lidar2image.tensor", frame_dir.c_str()), false);
  auto img_aug_matrix = nv::Tensor::load(nv::format("%s/img_aug_matrix.tensor", frame_dir.c_str()), false);

  core_->update(camera2lidar.ptr<float>(),
                camera_intrinsics.ptr<float>(),
                lidar2image.ptr<float>(),
                img_aug_matrix.ptr<float>(),
                stream_);

  auto images = load_camera_images(frame_dir, true);
  auto lidar_points = nv::Tensor::load(nv::format("%s/points.tensor", frame_dir.c_str()), false);

  // CameraImages 内部保存的是 stbi_load 返回的 unsigned char*。
  // core_->forward() 只读取图像数据，不会修改图像，因此这里显式构造 const 指针数组。
  // 不能把 unsigned char** 直接强转成 const unsigned char**，否则 C++ 会认为存在 const 安全问题。
  std::vector<const unsigned char*> image_ptrs;
  image_ptrs.reserve(images.data.size());
  for (auto* image : images.data) {
    image_ptrs.push_back(static_cast<const unsigned char*>(image));
  }

  result.bboxes = core_->forward(image_ptrs.data(),
                                 lidar_points.ptr<nvtype::half>(),
                                 static_cast<int>(lidar_points.size(0)),
                                 stream_);

  // 关键修复：
  // 之前这里只返回 bbox，没有生成 result.visualization_image。
  // 节点里的保存和发布逻辑都依赖 result.visualization_image.empty() 判断，
  // 所以不渲染就不会保存 jpg，也不会发布 ROS2 Image。
  if (config_.render_visualization_image) {
    result.visualization_image = render_visualization(result.bboxes,
                                                      lidar_points,
                                                      images.data,
                                                      lidar2image,
                                                      stream_);
  }

  return result;
}

}  // namespace bevfusion_app
