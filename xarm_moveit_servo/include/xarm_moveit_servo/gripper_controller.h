/* Copyright 2025 UFACTORY Inc. All Rights Reserved.
 *
 * Software License Agreement (BSD License)
 *
 * Author: Vinman <vinman.cub@gmail.com>
 ============================================================================*/

#ifndef XARM_MOVEIT_SERVO_GRIPPER_CONTROLLER_H
#define XARM_MOVEIT_SERVO_GRIPPER_CONTROLLER_H

#include "dynamixel_sdk/dynamixel_sdk.h"
#include <mutex>
#include <string>

namespace xarm_moveit_servo {

class GripperController {
public:
  GripperController(const std::string &port_name, uint32_t baud_rate);
  ~GripperController();
  bool moveWithVelocity(int velocity_raw, int32_t &current_position);

  // Position Limits
  static const int32_t POS_MIN = 2600;
  static const int32_t POS_MAX = 3700;

private:
  // Dynamixel SDK handlers
  dynamixel::PortHandler *portHandler_;
  dynamixel::PacketHandler *packetHandler_;

  // Connection properties
  std::string port_name_;
  uint32_t baud_rate_;
  uint8_t dxl_id_ = 1;

  // Control Table Addresses
  const uint16_t ADDR_TORQUE_ENABLE = 64;
  const uint16_t ADDR_OPERATING_MODE = 11;
  const uint16_t ADDR_GOAL_VELOCITY = 104;
  const uint16_t ADDR_PRESENT_POSITION = 132;

  // Thread safety
  std::mutex mutex_;

  void log(const char *msg, int result, int error);
};

} // namespace xarm_moveit_servo

#endif // XARM_MOVEIT_SERVO_GRIPPER_CONTROLLER_H
