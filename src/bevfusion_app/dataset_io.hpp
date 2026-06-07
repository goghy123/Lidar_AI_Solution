#pragma once

#include <string>
#include <vector>

namespace bevfusion_app {

// 原项目固定使用 nuScenes mini 转换后的 6 相机顺序。
// 这里集中维护文件名，避免 ROS2 节点和 runner 中重复硬编码。
extern const char* kCameraImageFiles[6];

// 用 RAII 管理 stbi_load 返回的图像内存，避免某一帧异常时忘记释放。
class CameraImages {
 public:
  CameraImages() = default;
  ~CameraImages();

  CameraImages(const CameraImages&) = delete;
  CameraImages& operator=(const CameraImages&) = delete;

  CameraImages(CameraImages&& other) noexcept;
  CameraImages& operator=(CameraImages&& other) noexcept;

  std::vector<unsigned char*> data;
  std::vector<int> widths;
  std::vector<int> heights;
  std::vector<int> channels;
};

// 判断输入目录是否是 scene 根目录：只要下面存在 frame_xxxxxx 子目录，就认为是多帧 scene。
bool is_frame_root(const std::string& root);

// 支持两种输入：
// 1. data_dir 指向 scene-0，返回其中排序后的 frame_xxxxxx；
// 2. data_dir 直接指向 frame_000000，返回它本身。
std::vector<std::string> collect_frame_dirs(const std::string& root);

std::string basename_of_path(const std::string& path);

// ROS2 节点保存 jpg 前创建输出目录；如果目录已存在不会报错。
bool ensure_directory(const std::string& path, std::string* error_message = nullptr);

// 处理每帧前做文件完整性检查，避免 Tensor::load 或 stbi_load 在深层位置失败。
bool validate_frame_dir(const std::string& frame_dir, std::string* error_message = nullptr);

// 读取 6 张图像。这里强制 desired_channels=3，因为后续 CUDA 可视化代码假设 RGB 三通道。
CameraImages load_camera_images(const std::string& frame_dir, bool check_size = true);

}  // namespace bevfusion_app
