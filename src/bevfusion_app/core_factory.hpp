#pragma once

#include <memory>
#include <string>

#include "bevfusion/bevfusion.hpp"

namespace bevfusion_app {

struct CoreFactoryConfig {
  std::string model_root;
  std::string model_name = "resnet50int8";
  std::string precision = "int8";
  float confidence_threshold = 0.12f;
  bool sorted_bboxes = true;
  bool use_layernorm_plugin_plan = false;
};

// 通过绝对 model_root 创建 BEVFusion core，避免 ros2 run 时工作目录不同导致模型找不到。
std::shared_ptr<bevfusion::Core> create_core_from_config(const CoreFactoryConfig& config);

}  // namespace bevfusion_app
