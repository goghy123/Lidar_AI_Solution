#include "bevfusion_app/visualization_renderer.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "common/check.hpp"
#include "common/visualize.hpp"

namespace bevfusion_app {

RgbImage render_visualization(const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes,
                              const nv::Tensor& lidar_points,
                              const std::vector<unsigned char*>& images,
                              const nv::Tensor& lidar2image,
                              cudaStream_t stream) {
  if (images.size() != 6) {
    throw std::runtime_error("可视化需要 6 张相机图像，当前数量=" + std::to_string(images.size()));
  }

  // BoundingBox 和 nv::Prediction 字段布局一致，原 main.cpp 也是直接 memcpy。
  std::vector<nv::Prediction> predictions(bboxes.size());
  if (!bboxes.empty()) {
    std::memcpy(predictions.data(), bboxes.data(), bboxes.size() * sizeof(nv::Prediction));
  }

  const int padding = 300;
  const int lidar_size = 1024;
  const int content_width = lidar_size + padding * 3;
  const int content_height = 1080;

  nv::SceneArtistParameter scene_artist_param;
  scene_artist_param.width = content_width;
  scene_artist_param.height = content_height;
  scene_artist_param.stride = scene_artist_param.width * 3;

  nv::Tensor scene_device_image(std::vector<int>{scene_artist_param.height, scene_artist_param.width, 3}, nv::DataType::UInt8);
  scene_device_image.memset(0x00, stream);

  scene_artist_param.image_device = scene_device_image.ptr<unsigned char>();
  auto scene = nv::create_scene_artist(scene_artist_param);

  nv::BEVArtistParameter bev_artist_param;
  bev_artist_param.image_width = content_width;
  bev_artist_param.image_height = content_height;
  bev_artist_param.rotate_x = 70.0f;
  bev_artist_param.norm_size = lidar_size * 0.5f;
  bev_artist_param.cx = content_width * 0.5f;
  bev_artist_param.cy = content_height * 0.5f;
  bev_artist_param.image_stride = scene_artist_param.stride;

  // 点云 tensor 从文件加载在 host 上，这里复制到 device 后给 BEVArtist 绘制。
  auto points = lidar_points.to_device(stream);
  auto bev_visualizer = nv::create_bev_artist(bev_artist_param);
  bev_visualizer->draw_lidar_points(points.ptr<nvtype::half>(), points.size(0));
  bev_visualizer->draw_prediction(predictions, false);
  bev_visualizer->draw_ego();
  bev_visualizer->apply(scene_device_image.ptr<unsigned char>(), stream);

  nv::ImageArtistParameter image_artist_param;
  image_artist_param.num_camera = static_cast<int>(images.size());
  image_artist_param.image_width = 1600;
  image_artist_param.image_height = 900;
  image_artist_param.image_stride = image_artist_param.image_width * 3;
  image_artist_param.viewport_nx4x4.resize(images.size() * 4 * 4);
  std::memcpy(image_artist_param.viewport_nx4x4.data(), lidar2image.ptr<float>(),
              sizeof(float) * image_artist_param.viewport_nx4x4.size());

  const int gap = 0;
  const int camera_width = 500;
  const int camera_height = static_cast<float>(camera_width / static_cast<float>(image_artist_param.image_width) *
                                               image_artist_param.image_height);
  const int offset_cameras[][3] = {
      {-camera_width / 2, -content_height / 2 + gap, 0},
      {content_width / 2 - camera_width - gap, -content_height / 2 + camera_height / 2, 0},
      {-content_width / 2 + gap, -content_height / 2 + camera_height / 2, 0},
      {-camera_width / 2, +content_height / 2 - camera_height - gap, 1},
      {-content_width / 2 + gap, +content_height / 2 - camera_height - camera_height / 2, 0},
      {content_width / 2 - camera_width - gap, +content_height / 2 - camera_height - camera_height / 2, 1}};

  auto visualizer = nv::create_image_artist(image_artist_param);
  for (size_t icamera = 0; icamera < images.size(); ++icamera) {
    const int ox = offset_cameras[icamera][0] + content_width / 2;
    const int oy = offset_cameras[icamera][1] + content_height / 2;
    const bool xflip = static_cast<bool>(offset_cameras[icamera][2]);
    visualizer->draw_prediction(static_cast<int>(icamera), predictions, xflip);

    nv::Tensor device_image(std::vector<int>{900, 1600, 3}, nv::DataType::UInt8);
    device_image.copy_from_host(images[icamera], stream);

    if (xflip) {
      auto clone = device_image.clone(stream);
      scene->flipx(clone.ptr<unsigned char>(), clone.size(1), clone.size(1) * 3, clone.size(0),
                   device_image.ptr<unsigned char>(), device_image.size(1) * 3, stream);
      checkRuntime(cudaStreamSynchronize(stream));
    }

    visualizer->apply(device_image.ptr<unsigned char>(), stream);

    scene->resize_to(device_image.ptr<unsigned char>(), ox, oy, ox + camera_width, oy + camera_height,
                     device_image.size(1), device_image.size(1) * 3, device_image.size(0), 0.8f, stream);
    checkRuntime(cudaStreamSynchronize(stream));
  }

  // ROS2 Image 需要 host 端 RGB buffer。这里复制成 std::vector，确保函数返回后内存仍有效。
  auto host_image = scene_device_image.to_host(stream);
  checkRuntime(cudaStreamSynchronize(stream));

  RgbImage output;
  output.width = static_cast<int>(scene_device_image.size(1));
  output.height = static_cast<int>(scene_device_image.size(0));
  output.channels = 3;
  output.data.resize(static_cast<size_t>(output.width) * output.height * output.channels);
  std::memcpy(output.data.data(), host_image.ptr<unsigned char>(), output.data.size());

  return output;
}

bool save_rgb_image_as_jpeg(const std::string& save_path, const RgbImage& image, int quality) {
  if (image.empty()) {
    return false;
  }

  const int clamped_quality = std::max(1, std::min(100, quality));
  return stbi_write_jpg(save_path.c_str(), image.width, image.height, image.channels,
                        image.data.data(), clamped_quality) != 0;
}

}  // namespace bevfusion_app
