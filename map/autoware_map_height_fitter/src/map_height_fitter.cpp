// Copyright 2022 The Autoware Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "autoware/map_height_fitter/map_height_fitter.hpp"

#include <autoware_lanelet2_extension/utility/message_conversion.hpp>
#include <autoware_lanelet2_extension/utility/query.hpp>

#include <autoware_map_msgs/msg/lanelet_map_bin.hpp>
#include <autoware_map_msgs/srv/get_partial_point_cloud_map.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <lanelet2_core/Forward.h>
#include <lanelet2_core/LaneletMap.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_listener.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>

namespace autoware::map_height_fitter
{

struct MapHeightFitter::Impl
{
  static constexpr char enable_partial_load[] = "enable_partial_load";

  explicit Impl(rclcpp::Node * node);
  void on_pcd_map(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void on_vector_map(const autoware_map_msgs::msg::LaneletMapBin::ConstSharedPtr msg);
  bool get_partial_point_cloud_map(const Point & point);
  double get_ground_height(const Point & point) const;
  std::optional<Point> fit(const Point & position, const std::string & frame);

  tf2::BufferCore tf2_buffer_;
  tf2_ros::TransformListener tf2_listener_;
  std::string map_frame_;
  rclcpp::Node * node_;

  std::string fit_target_;

  // for fitting by pointcloud_map_loader
  rclcpp::CallbackGroup::SharedPtr group_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_;
  rclcpp::Client<autoware_map_msgs::srv::GetPartialPointCloudMap>::SharedPtr cli_pcd_map_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcd_map_;
  rclcpp::AsyncParametersClient::SharedPtr params_pcd_map_loader_;

  // for fitting by vector_map_loader
  lanelet::LaneletMapPtr vector_map_;
  rclcpp::Subscription<autoware_map_msgs::msg::LaneletMapBin>::SharedPtr sub_vector_map_;
};

MapHeightFitter::Impl::Impl(rclcpp::Node * node) : tf2_listener_(tf2_buffer_), node_(node)
{
  fit_target_ = node->declare_parameter<std::string>("map_height_fitter.target");
  RCLCPP_DEBUG(node_->get_logger(), "[MapHeightFitter] fit_target: %s", fit_target_.c_str());

  if (fit_target_ == "pointcloud_map") {
    const auto callback =
      [this](const std::shared_future<std::vector<rclcpp::Parameter>> & future) {
        bool partial_load = false;
        for (const auto & param : future.get()) {
          if (param.get_name() == enable_partial_load) {
            partial_load = param.as_bool();
          }
        }

        if (partial_load) {
          group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
          cli_pcd_map_ = node_->create_client<autoware_map_msgs::srv::GetPartialPointCloudMap>(
            "~/partial_map_load", rmw_qos_profile_default, group_);
        } else {
          const auto durable_qos = rclcpp::QoS(1).transient_local();
          sub_pcd_map_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            "~/pointcloud_map", durable_qos,
            std::bind(&MapHeightFitter::Impl::on_pcd_map, this, std::placeholders::_1));
        }
      };

    const auto map_loader_name =
      node->declare_parameter<std::string>("map_height_fitter.map_loader_name");
    params_pcd_map_loader_ = rclcpp::AsyncParametersClient::make_shared(node, map_loader_name);
    params_pcd_map_loader_->wait_for_service();
    params_pcd_map_loader_->get_parameters({enable_partial_load}, callback);

  } else if (fit_target_ == "vector_map") {
    const auto vm_topic = node_->declare_parameter<std::string>("map_height_fitter.vector_map_topic", "/map/vector_map");
    const auto durable_qos = rclcpp::QoS(1).transient_local();
    sub_vector_map_ = node_->create_subscription<autoware_map_msgs::msg::LaneletMapBin>(
      vm_topic, durable_qos,
      std::bind(&MapHeightFitter::Impl::on_vector_map, this, std::placeholders::_1));

  } else {
    throw std::runtime_error("invalid fit_target");
  }
}

void MapHeightFitter::Impl::on_pcd_map(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  map_frame_ = msg->header.frame_id;
  map_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::fromROSMsg(*msg, *map_cloud_);
}

bool MapHeightFitter::Impl::get_partial_point_cloud_map(const Point & point)
{
  const auto logger = node_->get_logger();

  if (!cli_pcd_map_) {
    RCLCPP_WARN_STREAM(logger, "Partial map loading in pointcloud_map_loader is not enabled");
    return false;
  }
  if (!cli_pcd_map_->service_is_ready()) {
    RCLCPP_WARN_STREAM(logger, "Partial map loading in pointcloud_map_loader is not ready");
    return false;
  }

  const auto req = std::make_shared<autoware_map_msgs::srv::GetPartialPointCloudMap::Request>();
  req->area.center_x = static_cast<float>(point.x);
  req->area.center_y = static_cast<float>(point.y);
  req->area.radius = 50.0f;
  RCLCPP_DEBUG(
    logger, "[MapHeightFitter] Requesting partial PCD around (x=%.3f, y=%.3f), r=%.1f",
    point.x, point.y, static_cast<double>(req->area.radius));

  auto future = cli_pcd_map_->async_send_request(req);
  auto status = future.wait_for(std::chrono::seconds(1));
  int loops = 0;
  while (status != std::future_status::ready) {
    if (!rclcpp::ok()) return false;
    ++loops;
    RCLCPP_DEBUG(logger, "[MapHeightFitter] Waiting partial map response... (%d)", loops);
    status = future.wait_for(std::chrono::seconds(1));
  }

  const auto res = future.get();
  RCLCPP_DEBUG(
    logger, "[MapHeightFitter] Partial map grids received: %lu",
    static_cast<unsigned long>(res->new_pointcloud_with_ids.size()));

  sensor_msgs::msg::PointCloud2 pcd_msg;
  for (const auto & pcd_with_id : res->new_pointcloud_with_ids) {
    if (pcd_msg.width == 0) {
      pcd_msg = pcd_with_id.pointcloud;
    } else {
      pcd_msg.width += pcd_with_id.pointcloud.width;
      pcd_msg.row_step += pcd_with_id.pointcloud.row_step;
      pcd_msg.data.insert(
        pcd_msg.data.end(), pcd_with_id.pointcloud.data.begin(), pcd_with_id.pointcloud.data.end());
    }
  }
  map_frame_ = res->header.frame_id;
  map_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::fromROSMsg(pcd_msg, *map_cloud_);
  return true;
}

void MapHeightFitter::Impl::on_vector_map(
  const autoware_map_msgs::msg::LaneletMapBin::ConstSharedPtr msg)
{
  vector_map_ = std::make_shared<lanelet::LaneletMap>();
  lanelet::utils::conversion::fromBinMsg(*msg, vector_map_);
  map_frame_ = msg->header.frame_id;

  const auto n_points = vector_map_ ? vector_map_->pointLayer.size() : 0UL;
  const auto n_lines  = vector_map_ ? vector_map_->lineStringLayer.size() : 0UL;
  const auto n_lanes  = vector_map_ ? vector_map_->laneletLayer.size() : 0UL;

  RCLCPP_DEBUG(
    node_->get_logger(),
    "[MapHeightFitter] Received vector map: frame=%s points=%lu lines=%lu lanelets=%lu",
    map_frame_.c_str(),
    static_cast<unsigned long>(n_points),
    static_cast<unsigned long>(n_lines),
    static_cast<unsigned long>(n_lanes));
}

double MapHeightFitter::Impl::get_ground_height(const Point & point) const
{
  const auto logger = node_->get_logger();
  const double x = point.x, y = point.y;

  double height = INFINITY;

  if (fit_target_ == "pointcloud_map") {
    if (!map_cloud_ || map_cloud_->empty()) return point.z;

    double min_dist2 = INFINITY;
    for (const auto & p : map_cloud_->points) {
      const double dx = x - p.x, dy = y - p.y;
      const double sd = dx * dx + dy * dy;
      min_dist2 = std::min(min_dist2, sd);
    }

    const double d = std::sqrt(min_dist2);
    const double radius2 = (d + 1.0) * (d + 1.0);

    for (const auto & p : map_cloud_->points) {
      const double dx = x - p.x, dy = y - p.y;
      const double sd = dx * dx + dy * dy;
      if (sd < radius2) height = std::min(height, static_cast<double>(p.z));
    }

    if (std::isfinite(height)) {
      RCLCPP_DEBUG(logger, "[MapHeightFitter] PCD: nearest_radius=%.3f z=%.3f", d + 1.0, height);
    } else {
      RCLCPP_DEBUG(logger, "[MapHeightFitter] PCD: no neighbor within radius; keep z=%.3f", point.z);
    }
  } else if (fit_target_ == "vector_map") {
    if (!vector_map_) return point.z;
    const auto nearest = vector_map_->pointLayer.nearest(lanelet::BasicPoint2d{x, y}, 1);
    if (nearest.empty()) {
      RCLCPP_DEBUG(logger, "[MapHeightFitter] VectorMap: no nearby points; keep z=%.3f", point.z);
      return point.z;
    }
    height = nearest.front().z();
    RCLCPP_DEBUG(
      logger, "[MapHeightFitter] VectorMap: nearest point z=%.3f (x=%.3f y=%.3f)",
      height, nearest.front().x(), nearest.front().y());
  }

  return std::isfinite(height) ? height : point.z;
}

std::optional<Point>
MapHeightFitter::Impl::fit(const Point & position, const std::string & frame)
{
  const auto logger = node_->get_logger();
  RCLCPP_DEBUG(
    logger, "[MapHeightFitter] fit() called: pos=(%.3f, %.3f, %.3f) src_frame=%s",
    position.x, position.y, position.z, frame.c_str());
  RCLCPP_DEBUG(logger, "[MapHeightFitter] Current map_frame_='%s'", map_frame_.c_str());

  Point point = position;

  if (fit_target_ == "pointcloud_map") {
    if (cli_pcd_map_) {
      if (!get_partial_point_cloud_map(position)) {
        RCLCPP_DEBUG(logger, "[MapHeightFitter] Failed to get partial PCD");
        return std::nullopt;
      }
    }
    if (!map_cloud_) {
      RCLCPP_DEBUG(logger, "[MapHeightFitter] map_cloud_ is null");
      return std::nullopt;
    }
  } else if (fit_target_ == "vector_map") {
    if (!vector_map_) {
      RCLCPP_DEBUG(logger, "[MapHeightFitter] vector_map_ not ready");
      return std::nullopt;
    }
  } else {
    throw std::runtime_error("invalid fit_target");
  }

  try {
    const auto stamped = tf2_buffer_.lookupTransform(frame, map_frame_, tf2::TimePointZero);
    tf2::doTransform(point, point, stamped);
  } catch (tf2::TransformException & ex) {
    RCLCPP_DEBUG(logger, "[MapHeightFitter] TF %s->%s failed: %s",
                frame.c_str(), map_frame_.c_str(), ex.what());
    return std::nullopt;
  }

  const double old_z = point.z;
  point.z = get_ground_height(point);

  try {
    const auto stamped = tf2_buffer_.lookupTransform(map_frame_, frame, tf2::TimePointZero);
    tf2::doTransform(point, point, stamped);
  } catch (tf2::TransformException & ex) {
    RCLCPP_DEBUG(logger, "[MapHeightFitter] TF %s->%s failed: %s",
                map_frame_.c_str(), frame.c_str(), ex.what());
    return std::nullopt;
  }

  RCLCPP_DEBUG(logger, "[MapHeightFitter] Height fitted: z %.3f -> %.3f", old_z, point.z);
  return point;
}

MapHeightFitter::MapHeightFitter(rclcpp::Node * node)
{
  impl_ = std::make_unique<Impl>(node);
}

MapHeightFitter::~MapHeightFitter() = default;

std::optional<Point>
MapHeightFitter::fit(const Point & position, const std::string & frame)
{
  return impl_->fit(position, frame);
}

}  // namespace autoware::map_height_fitter