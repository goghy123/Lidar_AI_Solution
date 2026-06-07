#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "bevfusion/bevfusion.hpp"
#include "bevfusion_app/core_factory.hpp"
#include "bevfusion_app/visualization_renderer.hpp"

namespace bevfusion_app {

struct RunnerConfig {
  CoreFactoryConfig core;
  bool enable_core_timer = true;
  bool render_visualization_image = true;
};

struct FrameResult {
  std::string frame_dir;
  std::string frame_name;
  std::vector<bevfusion::head::transbbox::BoundingBox> bboxes;
  RgbImage visualization_image;
};

class BevFusionRunner {
 public:
  explicit BevFusionRunner(const RunnerConfig& config);
  ~BevFusionRunner();

  BevFusionRunner(const BevFusionRunner&) = delete;
  BevFusionRunner& operator=(const BevFusionRunner&) = delete;

  FrameResult process_frame(const std::string& frame_dir);

 private:
  RunnerConfig config_;
  std::shared_ptr<bevfusion::Core> core_;
  cudaStream_t stream_ = nullptr;
};

}  // namespace bevfusion_app
