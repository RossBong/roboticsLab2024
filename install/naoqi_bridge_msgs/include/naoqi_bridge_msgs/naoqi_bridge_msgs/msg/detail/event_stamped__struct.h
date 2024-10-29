// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from naoqi_bridge_msgs:msg/EventStamped.idl
// generated code does not contain a copyright notice

#ifndef NAOQI_BRIDGE_MSGS__MSG__DETAIL__EVENT_STAMPED__STRUCT_H_
#define NAOQI_BRIDGE_MSGS__MSG__DETAIL__EVENT_STAMPED__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'name'
// Member 'data'
#include "std_msgs/msg/detail/string__struct.h"

/// Struct defined in msg/EventStamped in the package naoqi_bridge_msgs.
typedef struct naoqi_bridge_msgs__msg__EventStamped
{
  std_msgs__msg__Header header;
  std_msgs__msg__String name;
  std_msgs__msg__String data;
} naoqi_bridge_msgs__msg__EventStamped;

// Struct for a sequence of naoqi_bridge_msgs__msg__EventStamped.
typedef struct naoqi_bridge_msgs__msg__EventStamped__Sequence
{
  naoqi_bridge_msgs__msg__EventStamped * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} naoqi_bridge_msgs__msg__EventStamped__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // NAOQI_BRIDGE_MSGS__MSG__DETAIL__EVENT_STAMPED__STRUCT_H_