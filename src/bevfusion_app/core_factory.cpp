#include "bevfusion_app/core_factory.hpp"

#include <filesystem>
#include <stdexcept>

namespace bevfusion_app {
namespace {

std::string join_path(const std::filesystem::path& base, const std::string& child) {
  return (base / child).string();
}

}  // namespace

std::shared_ptr<bevfusion::Core> create_core_from_config(const CoreFactoryConfig& config) {
  namespace fs = std::filesystem;

  if (config.model_root.empty()) {
    throw std::runtime_error("model_root 为空，请在 YAML 中配置模型根目录");
  }

  const fs::path model_base = fs::path(config.model_root) / config.model_name;
  const fs::path build_dir = model_base / "build";

  printf("Create BEVFusion core: model_root=%s, model_name=%s, precision=%s\n",
         config.model_root.c_str(), config.model_name.c_str(), config.precision.c_str());

  bevfusion::camera::NormalizationParameter normalization;
  normalization.image_width = 1600;
  normalization.image_height = 900;
  normalization.output_width = 704;
  normalization.output_height = 256;
  normalization.num_camera = 6;
  normalization.resize_lim = 0.48f;
  normalization.interpolation = bevfusion::camera::Interpolation::Bilinear;

  float mean[3] = {0.485, 0.456, 0.406};
  float std[3] = {0.229, 0.224, 0.225};
  normalization.method = bevfusion::camera::NormMethod::mean_std(mean, std, 1 / 255.0f, 0.0f);

  bevfusion::lidar::VoxelizationParameter voxelization;
  voxelization.min_range = nvtype::Float3(-54.0f, -54.0f, -5.0);
  voxelization.max_range = nvtype::Float3(+54.0f, +54.0f, +3.0);
  voxelization.voxel_size = nvtype::Float3(0.075f, 0.075f, 0.2f);
  voxelization.grid_size =
      voxelization.compute_grid_size(voxelization.max_range, voxelization.min_range, voxelization.voxel_size);
  voxelization.max_points_per_voxel = 10;
  voxelization.max_points = 300000;
  voxelization.max_voxels = 160000;
  voxelization.num_feature = 5;

  bevfusion::lidar::SCNParameter scn;
  scn.voxelization = voxelization;
  scn.model = join_path(model_base, "lidar.backbone.xyz.onnx");
  scn.order = bevfusion::lidar::CoordinateOrder::XYZ;

  if (config.precision == "int8") {
    scn.precision = bevfusion::lidar::Precision::Int8;
  } else {
    scn.precision = bevfusion::lidar::Precision::Float16;
  }

  bevfusion::camera::GeometryParameter geometry;
  geometry.xbound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
  geometry.ybound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
  geometry.zbound = nvtype::Float3(-10.0f, 10.0f, 20.0f);
  geometry.dbound = nvtype::Float3(1.0, 60.0f, 0.5f);
  geometry.image_width = 704;
  geometry.image_height = 256;
  geometry.feat_width = 88;
  geometry.feat_height = 32;
  geometry.num_camera = 6;
  geometry.geometry_dim = nvtype::Int3(360, 360, 80);

  bevfusion::head::transbbox::TransBBoxParameter transbbox;
  transbbox.out_size_factor = 8;
  transbbox.pc_range = {-54.0f, -54.0f};
  transbbox.post_center_range_start = {-61.2, -61.2, -10.0};
  transbbox.post_center_range_end = {61.2, 61.2, 10.0};
  transbbox.voxel_size = {0.075, 0.075};

  // 有些环境下普通 head.bbox.plan 的框会有精度问题，原项目也提示可以切换 layernormplugin 版本。
  transbbox.model = join_path(build_dir,
                              config.use_layernorm_plugin_plan ? "head.bbox.layernormplugin.plan" : "head.bbox.plan");
  transbbox.confidence_threshold = config.confidence_threshold;
  transbbox.sorted_bboxes = config.sorted_bboxes;

  bevfusion::CoreParameter param;
  param.camera_model = join_path(build_dir, "camera.backbone.plan");
  param.normalize = normalization;
  param.lidar_scn = scn;
  param.geometry = geometry;
  param.transfusion = join_path(build_dir, "fuser.plan");
  param.transbbox = transbbox;
  param.camera_vtransform = join_path(build_dir, "camera.vtransform.plan");

  return bevfusion::create_core(param);
}

}  // namespace bevfusion_app
