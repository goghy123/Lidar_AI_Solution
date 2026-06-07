#include "bevfusion_app/dataset_io.hpp"

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <utility>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace bevfusion_app {

const char* kCameraImageFiles[6] = {
    "0-FRONT.jpg",       "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
    "3-BACK.jpg",        "4-BACK_LEFT.jpg",   "5-BACK_RIGHT.jpg"};

CameraImages::~CameraImages() {
  for (auto* ptr : data) {
    if (ptr != nullptr) {
      stbi_image_free(ptr);
    }
  }
}

CameraImages::CameraImages(CameraImages&& other) noexcept
    : data(std::move(other.data)),
      widths(std::move(other.widths)),
      heights(std::move(other.heights)),
      channels(std::move(other.channels)) {
  other.data.clear();
}

CameraImages& CameraImages::operator=(CameraImages&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  for (auto* ptr : data) {
    if (ptr != nullptr) {
      stbi_image_free(ptr);
    }
  }

  data = std::move(other.data);
  widths = std::move(other.widths);
  heights = std::move(other.heights);
  channels = std::move(other.channels);
  other.data.clear();
  return *this;
}

bool is_frame_root(const std::string& root) {
  namespace fs = std::filesystem;

  if (!fs::exists(root) || !fs::is_directory(root)) {
    return false;
  }

  for (const auto& p : fs::directory_iterator(root)) {
    if (!p.is_directory()) {
      continue;
    }

    const auto name = p.path().filename().string();
    if (name.rfind("frame_", 0) == 0) {
      return true;
    }
  }

  return false;
}

std::vector<std::string> collect_frame_dirs(const std::string& root) {
  namespace fs = std::filesystem;

  if (!fs::exists(root) || !fs::is_directory(root)) {
    throw std::runtime_error("data_dir 不存在或不是目录: " + root);
  }

  std::vector<std::string> frames;
  if (is_frame_root(root)) {
    for (const auto& p : fs::directory_iterator(root)) {
      if (!p.is_directory()) {
        continue;
      }

      const auto name = p.path().filename().string();
      if (name.rfind("frame_", 0) == 0) {
        frames.push_back(p.path().string());
      }
    }

    std::sort(frames.begin(), frames.end());
  } else {
    frames.push_back(root);
  }

  if (frames.empty()) {
    throw std::runtime_error("没有找到可处理的 frame 目录: " + root);
  }

  return frames;
}

std::string basename_of_path(const std::string& path) {
  return std::filesystem::path(path).filename().string();
}

bool ensure_directory(const std::string& path, std::string* error_message) {
  namespace fs = std::filesystem;

  try {
    if (path.empty()) {
      if (error_message != nullptr) {
        *error_message = "输出目录为空";
      }
      return false;
    }

    fs::create_directories(path);
    return true;
  } catch (const std::exception& e) {
    if (error_message != nullptr) {
      *error_message = e.what();
    }
    return false;
  }
}

bool validate_frame_dir(const std::string& frame_dir, std::string* error_message) {
  namespace fs = std::filesystem;

  if (!fs::exists(frame_dir) || !fs::is_directory(frame_dir)) {
    if (error_message != nullptr) {
      *error_message = "frame 目录不存在: " + frame_dir;
    }
    return false;
  }

  std::vector<std::string> required_files;
  for (const auto* name : kCameraImageFiles) {
    required_files.emplace_back(name);
  }

  required_files.emplace_back("camera2lidar.tensor");
  required_files.emplace_back("camera_intrinsics.tensor");
  required_files.emplace_back("lidar2image.tensor");
  required_files.emplace_back("img_aug_matrix.tensor");
  required_files.emplace_back("points.tensor");

  std::ostringstream missing;
  bool ok = true;
  for (const auto& file : required_files) {
    const auto path = fs::path(frame_dir) / file;
    if (!fs::exists(path)) {
      ok = false;
      missing << "  - " << path.string() << "\n";
    }
  }

  if (!ok && error_message != nullptr) {
    *error_message = "frame 文件不完整，缺少:\n" + missing.str();
  }

  return ok;
}

CameraImages load_camera_images(const std::string& frame_dir, bool check_size) {
  namespace fs = std::filesystem;

  CameraImages images;
  images.data.reserve(6);
  images.widths.reserve(6);
  images.heights.reserve(6);
  images.channels.reserve(6);

  for (int i = 0; i < 6; ++i) {
    const auto path = fs::path(frame_dir) / kCameraImageFiles[i];

    int width = 0;
    int height = 0;
    int source_channels = 0;

    // 强制输出 RGB 三通道，和原 CUDA 可视化函数的内存布局保持一致。
    unsigned char* image = stbi_load(path.string().c_str(), &width, &height, &source_channels, 3);
    if (image == nullptr) {
      throw std::runtime_error("读取相机图像失败: " + path.string());
    }

    if (check_size && (width != 1600 || height != 900)) {
      stbi_image_free(image);
      throw std::runtime_error("相机图像尺寸不是 1600x900: " + path.string() +
                               ", 实际尺寸=" + std::to_string(width) + "x" + std::to_string(height));
    }

    images.data.push_back(image);
    images.widths.push_back(width);
    images.heights.push_back(height);
    images.channels.push_back(3);
  }

  return images;
}

}  // namespace bevfusion_app
