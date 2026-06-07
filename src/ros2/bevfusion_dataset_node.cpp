#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "bevfusion_app/bevfusion_runner.hpp"
#include "bevfusion_app/dataset_io.hpp"
#include "bevfusion_app/visualization_renderer.hpp"

namespace {

struct NodeConfig {
  std::string data_dir;
  std::string model_root;
  std::string output_dir;

  std::string model_name;
  std::string precision;
  double confidence_threshold = 0.12;
  bool sorted_bboxes = true;
  bool use_layernorm_plugin_plan = false;
  std::string plugin_library = "libcustom_layernorm.so";

  std::string playback_mode = "timer";
  double publish_rate_hz = 5.0;
  bool loop = false;
  int start_frame_index = 0;
  int max_frames = -1;
  double start_delay_sec = 1.0;

  bool publish_visualization_image = true;
  std::string visualization_image_topic = "/bevfusion/visualization/image";
  std::string visualization_image_frame_id = "bevfusion_debug";

  bool save_visualization_image = false;
  int jpeg_quality = 100;

  bool print_bbox_summary = true;
  int print_top_k_bboxes = 5;

  bool enable_core_timer = true;
  bool validate_frame_files = true;
  bool skip_bad_frames = false;
};

class BevFusionDatasetNode : public rclcpp::Node {
 public:
  BevFusionDatasetNode() : Node("bevfusion_dataset_node") {
    read_parameters();
    validate_parameters();

    load_plugin_library();

    if (config_.publish_visualization_image) {
      // 调试图像只需要保留最新一帧；可靠传输可以避免 RViz2/rqt_image_view 偶尔收不到首帧。
      image_pub_ = create_publisher<sensor_msgs::msg::Image>(
          config_.visualization_image_topic, rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    }

    frame_dirs_ = bevfusion_app::collect_frame_dirs(config_.data_dir);
    apply_frame_range();

    if (config_.save_visualization_image) {
      std::string error;
      if (!bevfusion_app::ensure_directory(config_.output_dir, &error)) {
        throw std::runtime_error("创建 output_dir 失败: " + error);
      }
    }

    bevfusion_app::RunnerConfig runner_config;
    runner_config.core.model_root = config_.model_root;
    runner_config.core.model_name = config_.model_name;
    runner_config.core.precision = config_.precision;
    runner_config.core.confidence_threshold = static_cast<float>(config_.confidence_threshold);
    runner_config.core.sorted_bboxes = config_.sorted_bboxes;
    runner_config.core.use_layernorm_plugin_plan = config_.use_layernorm_plugin_plan;
    runner_config.enable_core_timer = config_.enable_core_timer;
    runner_config.render_visualization_image = config_.publish_visualization_image || config_.save_visualization_image;

    runner_ = std::make_unique<bevfusion_app::BevFusionRunner>(runner_config);

    RCLCPP_INFO(get_logger(), "数据集帧数: %zu", frame_dirs_.size());
    RCLCPP_INFO(get_logger(), "播放模式: %s", config_.playback_mode.c_str());
    RCLCPP_INFO(get_logger(), "图像话题: %s", config_.visualization_image_topic.c_str());

    // 给 RViz2/rqt_image_view 和其他节点一点启动时间，避免节点刚启动就发完第一帧。
    start_timer_ = create_wall_timer(
        std::chrono::duration<double>(std::max(0.0, config_.start_delay_sec)),
        std::bind(&BevFusionDatasetNode::start_playback, this));
  }

  ~BevFusionDatasetNode() override {
    if (plugin_handle_ != nullptr) {
      dlclose(plugin_handle_);
      plugin_handle_ = nullptr;
    }
  }

 private:
  void read_parameters() {
    config_.data_dir = declare_parameter<std::string>("data_dir", "example-data");
    config_.model_root = declare_parameter<std::string>("model_root", "model");
    config_.output_dir = declare_parameter<std::string>("output_dir", "build");

    config_.model_name = declare_parameter<std::string>("model_name", "resnet50int8");
    config_.precision = declare_parameter<std::string>("precision", "int8");
    config_.confidence_threshold = declare_parameter<double>("confidence_threshold", 0.12);
    config_.sorted_bboxes = declare_parameter<bool>("sorted_bboxes", true);
    config_.use_layernorm_plugin_plan = declare_parameter<bool>("use_layernorm_plugin_plan", false);
    config_.plugin_library = declare_parameter<std::string>("plugin_library", "libcustom_layernorm.so");

    config_.playback_mode = declare_parameter<std::string>("playback_mode", "timer");
    config_.publish_rate_hz = declare_parameter<double>("publish_rate_hz", 5.0);
    config_.loop = declare_parameter<bool>("loop", false);
    config_.start_frame_index = declare_parameter<int>("start_frame_index", 0);
    config_.max_frames = declare_parameter<int>("max_frames", -1);
    config_.start_delay_sec = declare_parameter<double>("start_delay_sec", 1.0);

    config_.publish_visualization_image = declare_parameter<bool>("publish_visualization_image", true);
    config_.visualization_image_topic =
        declare_parameter<std::string>("visualization_image_topic", "/bevfusion/visualization/image");
    config_.visualization_image_frame_id =
        declare_parameter<std::string>("visualization_image_frame_id", "bevfusion_debug");

    config_.save_visualization_image = declare_parameter<bool>("save_visualization_image", false);
    config_.jpeg_quality = declare_parameter<int>("jpeg_quality", 100);

    config_.print_bbox_summary = declare_parameter<bool>("print_bbox_summary", true);
    config_.print_top_k_bboxes = declare_parameter<int>("print_top_k_bboxes", 5);

    config_.enable_core_timer = declare_parameter<bool>("enable_core_timer", true);
    config_.validate_frame_files = declare_parameter<bool>("validate_frame_files", true);
    config_.skip_bad_frames = declare_parameter<bool>("skip_bad_frames", false);

    // 第一阶段暂不实现 Detection3DArray / MarkerArray，但允许 YAML 中保留这些预留参数。
    declare_parameter<bool>("publish_detections", false);
    declare_parameter<std::string>("detections_topic", "/bevfusion/detections");
    declare_parameter<std::string>("detections_frame_id", "lidar");
    declare_parameter<bool>("publish_markers", false);
    declare_parameter<std::string>("markers_topic", "/bevfusion/markers");
    declare_parameter<std::string>("markers_frame_id", "lidar");
  }

  void validate_parameters() const {
    if (config_.playback_mode != "timer" && config_.playback_mode != "immediate") {
      throw std::runtime_error("playback_mode 只能是 timer 或 immediate，当前值: " + config_.playback_mode);
    }

    if (config_.playback_mode == "timer" && config_.publish_rate_hz <= 0.0) {
      throw std::runtime_error("timer 模式下 publish_rate_hz 必须大于 0");
    }

    if (!config_.publish_visualization_image && !config_.save_visualization_image) {
      RCLCPP_WARN(get_logger(), "publish_visualization_image 和 save_visualization_image 都为 false，节点只会打印 bbox 信息");
    }
  }

  void load_plugin_library() {
    if (config_.plugin_library.empty()) {
      RCLCPP_WARN(get_logger(), "plugin_library 为空，跳过 dlopen");
      return;
    }

    plugin_handle_ = dlopen(config_.plugin_library.c_str(), RTLD_NOW);
    if (plugin_handle_ == nullptr) {
      // 原 demo 没有检查 dlopen 结果。这里先警告而不直接退出，避免不使用 layernormplugin plan 时过度阻塞。
      const char* error = dlerror();
      RCLCPP_WARN(get_logger(), "加载插件库失败: %s, dlerror=%s", config_.plugin_library.c_str(), error ? error : "unknown");
    } else {
      RCLCPP_INFO(get_logger(), "已加载插件库: %s", config_.plugin_library.c_str());
    }
  }

  void apply_frame_range() {
    if (frame_dirs_.empty()) {
      throw std::runtime_error("没有可处理的数据帧");
    }

    const int start = std::max(0, config_.start_frame_index);
    if (start >= static_cast<int>(frame_dirs_.size())) {
      throw std::runtime_error("start_frame_index 超出数据集帧数");
    }

    auto begin = frame_dirs_.begin() + start;
    auto end = frame_dirs_.end();
    if (config_.max_frames > 0) {
      end = begin + std::min<int>(config_.max_frames, static_cast<int>(std::distance(begin, frame_dirs_.end())));
    }

    frame_dirs_ = std::vector<std::string>(begin, end);
  }

  void start_playback() {
    if (start_timer_) {
      start_timer_->cancel();
    }

    if (config_.playback_mode == "timer") {
      const auto period = std::chrono::duration<double>(1.0 / config_.publish_rate_hz);
      playback_timer_ = create_wall_timer(period, std::bind(&BevFusionDatasetNode::process_one_frame, this));
      RCLCPP_INFO(get_logger(), "timer 模式启动，目标频率 %.3f Hz", config_.publish_rate_hz);
    } else {
      RCLCPP_INFO(get_logger(), "immediate 模式启动，将尽快处理数据集");
      while (rclcpp::ok()) {
        if (!process_one_frame()) {
          break;
        }
      }
    }
  }

  bool process_one_frame() {
    if (processing_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "上一帧仍在处理，跳过本次 timer 回调但不跳数据帧");
      return true;
    }

    processing_ = true;
    const auto reset_processing = [&]() { processing_ = false; };

    while (rclcpp::ok()) {
      if (frame_index_ >= frame_dirs_.size()) {
        if (config_.loop) {
          frame_index_ = 0;
          RCLCPP_INFO(get_logger(), "loop=true，重新从第一帧开始");
        } else {
          RCLCPP_INFO(get_logger(), "数据集处理完成");
          stop_playback();
          reset_processing();
          return false;
        }
      }

      const auto frame_dir = frame_dirs_[frame_index_++];
      const auto frame_name = bevfusion_app::basename_of_path(frame_dir);

      if (config_.validate_frame_files) {
        std::string error;
        if (!bevfusion_app::validate_frame_dir(frame_dir, &error)) {
          if (config_.skip_bad_frames) {
            RCLCPP_WARN(get_logger(), "跳过坏帧 %s: %s", frame_dir.c_str(), error.c_str());
            continue;
          }

          RCLCPP_ERROR(get_logger(), "坏帧导致节点停止 %s: %s", frame_dir.c_str(), error.c_str());
          stop_playback();
          reset_processing();
          return false;
        }
      }

      try {
        RCLCPP_INFO(get_logger(), "[%zu / %zu] 处理 %s", frame_index_, frame_dirs_.size(), frame_dir.c_str());
        auto result = runner_->process_frame(frame_dir);

        print_bbox_summary(result);

        RCLCPP_INFO(get_logger(),
            "可视化图像状态: frame=%s, width=%d, height=%d, channels=%d, data=%zu, save=%d, publish=%d",
            result.frame_name.c_str(),
            result.visualization_image.width,
            result.visualization_image.height,
            result.visualization_image.channels,
            result.visualization_image.data.size(),
            config_.save_visualization_image,
            config_.publish_visualization_image);


        if (config_.save_visualization_image && !result.visualization_image.empty()) {
          save_visualization(result);
        }

        if (config_.publish_visualization_image && !result.visualization_image.empty()) {
          publish_visualization(result.visualization_image);
        }
      } catch (const std::exception& e) {
        if (config_.skip_bad_frames) {
          RCLCPP_WARN(get_logger(), "处理帧失败，已跳过 %s: %s", frame_dir.c_str(), e.what());
          continue;
        }

        RCLCPP_ERROR(get_logger(), "处理帧失败，节点停止 %s: %s", frame_dir.c_str(), e.what());
        stop_playback();
        reset_processing();
        return false;
      }

      reset_processing();
      return true;
    }

    reset_processing();
    return false;
  }

  void stop_playback() {
    if (playback_timer_) {
      playback_timer_->cancel();
    }
  }

  void publish_visualization(const bevfusion_app::RgbImage& image) {
    if (!image_pub_) {
      return;
    }

    sensor_msgs::msg::Image msg;
    msg.header.stamp = now();
    msg.header.frame_id = config_.visualization_image_frame_id;
    msg.height = static_cast<uint32_t>(image.height);
    msg.width = static_cast<uint32_t>(image.width);
    msg.encoding = sensor_msgs::image_encodings::RGB8;
    msg.is_bigendian = false;
    msg.step = static_cast<sensor_msgs::msg::Image::_step_type>(image.width * image.channels);
    msg.data = image.data;

    image_pub_->publish(msg);
  }

  void save_visualization(const bevfusion_app::FrameResult& result) {
    namespace fs = std::filesystem;

    const auto save_path = (fs::path(config_.output_dir) / (result.frame_name + ".jpg")).string();
    if (bevfusion_app::save_rgb_image_as_jpeg(save_path, result.visualization_image, config_.jpeg_quality)) {
      RCLCPP_INFO(get_logger(), "保存可视化图片: %s", save_path.c_str());
    } else {
      RCLCPP_WARN(get_logger(), "保存可视化图片失败: %s", save_path.c_str());
    }
  }

  void print_bbox_summary(const bevfusion_app::FrameResult& result) const {
    if (!config_.print_bbox_summary) {
      return;
    }

    RCLCPP_INFO(get_logger(), "frame=%s, bbox 数量=%zu", result.frame_name.c_str(), result.bboxes.size());

    const int top_k = std::max(0, config_.print_top_k_bboxes);
    const int count = std::min<int>(top_k, static_cast<int>(result.bboxes.size()));
    for (int i = 0; i < count; ++i) {
      const auto& box = result.bboxes[i];
      RCLCPP_INFO(get_logger(),
                  "  bbox[%d]: id=%d score=%.4f pos=(%.2f, %.2f, %.2f) size=(%.2f, %.2f, %.2f) yaw=%.3f",
                  i,
                  box.id,
                  box.score,
                  box.position.x,
                  box.position.y,
                  box.position.z,
                  box.size.w,
                  box.size.l,
                  box.size.h,
                  box.z_rotation);
    }
  }

 private:
  NodeConfig config_;
  void* plugin_handle_ = nullptr;

  std::vector<std::string> frame_dirs_;
  size_t frame_index_ = 0;
  bool processing_ = false;

  std::unique_ptr<bevfusion_app::BevFusionRunner> runner_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::TimerBase::SharedPtr start_timer_;
  rclcpp::TimerBase::SharedPtr playback_timer_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<BevFusionDatasetNode>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    fprintf(stderr, "bevfusion_dataset_node 启动失败: %s\n", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
