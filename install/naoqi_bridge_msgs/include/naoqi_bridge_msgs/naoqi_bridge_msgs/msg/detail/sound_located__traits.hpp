// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from naoqi_bridge_msgs:msg/SoundLocated.idl
// generated code does not contain a copyright notice

#ifndef NAOQI_BRIDGE_MSGS__MSG__DETAIL__SOUND_LOCATED__TRAITS_HPP_
#define NAOQI_BRIDGE_MSGS__MSG__DETAIL__SOUND_LOCATED__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "naoqi_bridge_msgs/msg/detail/sound_located__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'head_position_frame_torso'
// Member 'head_position_frame_robot'
#include "geometry_msgs/msg/detail/twist__traits.hpp"

namespace naoqi_bridge_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const SoundLocated & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: azimuth
  {
    out << "azimuth: ";
    rosidl_generator_traits::value_to_yaml(msg.azimuth, out);
    out << ", ";
  }

  // member: elevation
  {
    out << "elevation: ";
    rosidl_generator_traits::value_to_yaml(msg.elevation, out);
    out << ", ";
  }

  // member: confidence
  {
    out << "confidence: ";
    rosidl_generator_traits::value_to_yaml(msg.confidence, out);
    out << ", ";
  }

  // member: energy
  {
    out << "energy: ";
    rosidl_generator_traits::value_to_yaml(msg.energy, out);
    out << ", ";
  }

  // member: head_position_frame_torso
  {
    out << "head_position_frame_torso: ";
    to_flow_style_yaml(msg.head_position_frame_torso, out);
    out << ", ";
  }

  // member: head_position_frame_robot
  {
    out << "head_position_frame_robot: ";
    to_flow_style_yaml(msg.head_position_frame_robot, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SoundLocated & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: azimuth
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "azimuth: ";
    rosidl_generator_traits::value_to_yaml(msg.azimuth, out);
    out << "\n";
  }

  // member: elevation
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "elevation: ";
    rosidl_generator_traits::value_to_yaml(msg.elevation, out);
    out << "\n";
  }

  // member: confidence
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "confidence: ";
    rosidl_generator_traits::value_to_yaml(msg.confidence, out);
    out << "\n";
  }

  // member: energy
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "energy: ";
    rosidl_generator_traits::value_to_yaml(msg.energy, out);
    out << "\n";
  }

  // member: head_position_frame_torso
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "head_position_frame_torso:\n";
    to_block_style_yaml(msg.head_position_frame_torso, out, indentation + 2);
  }

  // member: head_position_frame_robot
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "head_position_frame_robot:\n";
    to_block_style_yaml(msg.head_position_frame_robot, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SoundLocated & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace naoqi_bridge_msgs

namespace rosidl_generator_traits
{

[[deprecated("use naoqi_bridge_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const naoqi_bridge_msgs::msg::SoundLocated & msg,
  std::ostream & out, size_t indentation = 0)
{
  naoqi_bridge_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use naoqi_bridge_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const naoqi_bridge_msgs::msg::SoundLocated & msg)
{
  return naoqi_bridge_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<naoqi_bridge_msgs::msg::SoundLocated>()
{
  return "naoqi_bridge_msgs::msg::SoundLocated";
}

template<>
inline const char * name<naoqi_bridge_msgs::msg::SoundLocated>()
{
  return "naoqi_bridge_msgs/msg/SoundLocated";
}

template<>
struct has_fixed_size<naoqi_bridge_msgs::msg::SoundLocated>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::Twist>::value && has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<naoqi_bridge_msgs::msg::SoundLocated>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::Twist>::value && has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<naoqi_bridge_msgs::msg::SoundLocated>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // NAOQI_BRIDGE_MSGS__MSG__DETAIL__SOUND_LOCATED__TRAITS_HPP_