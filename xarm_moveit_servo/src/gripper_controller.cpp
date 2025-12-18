/* Copyright 2025 UFACTORY Inc. All Rights Reserved.
 *
 * Software License Agreement (BSD License)
 *
 * Author: Vinman <vinman.cub@gmail.com>
 ============================================================================*/

#include "xarm_moveit_servo/gripper_controller.h"
#include <iostream>
#include <thread>
#include <chrono>

namespace xarm_moveit_servo
{

GripperController::GripperController(const std::string& port_name, uint32_t baud_rate)
    : port_name_(port_name), baud_rate_(baud_rate)
{
    portHandler_ = dynamixel::PortHandler::getPortHandler(port_name_.c_str());
    packetHandler_ = dynamixel::PacketHandler::getPacketHandler(2.0); // Protocol 2.0

    // Open port
    if (!portHandler_->openPort()) {
        std::cerr << "Failed to open port " << port_name_ << std::endl;
        return;
    }
    std::cout << "Succeeded to open port " << port_name_ << std::endl;

    // Set baud rate
    if (!portHandler_->setBaudRate(baud_rate_)) {
        std::cerr << "Failed to set the baud rate to " << baud_rate_ << std::endl;
        return;
    }
    std::cout << "Succeeded to set the baud rate to " << baud_rate_ << std::endl;

    // Set to Velocity Control Mode
    uint8_t dxl_error = 0;
    int dxl_comm_result = packetHandler_->write1ByteTxRx(portHandler_, dxl_id_, ADDR_OPERATING_MODE, 1, &dxl_error);
    log("Set Velocity Control Mode", dxl_comm_result, dxl_error);

    // Enable Dynamixel Torque
    dxl_comm_result = packetHandler_->write1ByteTxRx(portHandler_, dxl_id_, ADDR_TORQUE_ENABLE, 1, &dxl_error);
    log("Enable Torque", dxl_comm_result, dxl_error);
}

GripperController::~GripperController()
{
    // Disable Dynamixel Torque
    uint8_t dxl_error = 0;
    int dxl_comm_result = packetHandler_->write1ByteTxRx(portHandler_, dxl_id_, ADDR_TORQUE_ENABLE, 0, &dxl_error);
    log("Disable Torque", dxl_comm_result, dxl_error);

    // Close port
    portHandler_->closePort();
}

bool GripperController::moveWithVelocity(int velocity_raw, int32_t& current_position)
{
    std::lock_guard<std::mutex> lock(mutex_);

    uint8_t dxl_error = 0;
    int dxl_comm_result = 0;

    // Read present position
    dxl_comm_result = packetHandler_->read4ByteTxRx(portHandler_, dxl_id_, ADDR_PRESENT_POSITION, (uint32_t*)&current_position, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS || dxl_error != 0) {
        log("Read Position", dxl_comm_result, dxl_error);
        return false;
    }

    // Add a delay to allow the half-duplex serial line to stabilize
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // Safety check
    int32_t velocity_to_set = velocity_raw;
    if ((current_position <= POS_MIN && velocity_raw < 0) || (current_position >= POS_MAX && velocity_raw > 0)) {
        velocity_to_set = 0;
    }

    // Write goal velocity
    dxl_comm_result = packetHandler_->write4ByteTxRx(portHandler_, dxl_id_, ADDR_GOAL_VELOCITY, velocity_to_set, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS || dxl_error != 0) {
        log("Write Goal Velocity", dxl_comm_result, dxl_error);
        return false;
    }

    return true;
}

void GripperController::log(const char* msg, int result, int error)
{
    if (result != COMM_SUCCESS) {
        std::cerr << msg << " failed: " << packetHandler_->getTxRxResult(result) << std::endl;
    } else if (error != 0) {
        std::cerr << msg << " error: " << packetHandler_->getRxPacketError(error) << std::endl;
    } else {
        std::cout << msg << " succeeded." << std::endl;
    }
}

} // namespace xarm_moveit_servo
