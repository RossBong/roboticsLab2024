// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from naoqi_bridge_msgs:msg/MemoryPairString.idl
// generated code does not contain a copyright notice

#ifndef NAOQI_BRIDGE_MSGS__MSG__DETAIL__MEMORY_PAIR_STRING__TRAITS_HPP_
#define NAOQI_BRIDGE_MSGS__MSG__DETAIL__MEMORY_PAIR_STRING__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "naoqi_bridge_msgs/msg/detail/memory_pair_string__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace naoqi_bridge_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const MemoryPairString & msg,
  std::ostream & out)
{
  out << "{";
  // member: memory_key
  {
    out << "memory_key: ";
    rosidl_generator_traits::value_to_yaml(msg.memory_key, out);
    out << ", ";
  }

  // member: data
  {
    out << "data: ";
    rosidl_generator_traits::value_to_yaml(msg.data, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const MemoryPairString & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: memory_key
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "memory_key: ";
    rosidl_generator_traits::value_to_yaml(msg.memory_key, out);
    out << "\n";
  }

  // member: data
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "data: ";
    rosidl_generator_traits::value_to_yaml(msg.data, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MemoryPairString & msg, bool use_flow_style = false)
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
  const naoqi_bridge_msgs::msg::MemoryPairString & msg,
  std::ostream & out, size_t indentation = 0)
{
  naoqi_bridge_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use naoqi_bridge_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const naoqi_bridge_msgs::msg::MemoryPairString & msg)
{
  return naoqi_bridge_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<naoqi_bridge_msgs::msg::MemoryPairString>()
{
  return "naoqi_bridge_msgs::msg::MemoryPairString";
}

template<>
inline const char * name<naoqi_bridge_msgs::msg::MemoryPairString>()
{
  return "naoqi_bridge_msgs/msg/MemoryPairString";
}

template<>
struct has_fixed_size<naoqi_bridge_msgs::msg::MemoryPairString>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<naoqi_bridge_msgs::msg::MemoryPairString>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<naoqi_bridge_msgs::msg::MemoryPairString>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // NAOQI_BRIDGE_MSGS__MSG__DETAIL__MEMORY_PAIR_STRING__TRAITS_HPP_