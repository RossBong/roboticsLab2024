// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from naoqi_bridge_msgs:msg/BodyROI.idl
// generated code does not contain a copyright notice

#ifndef NAOQI_BRIDGE_MSGS__MSG__DETAIL__BODY_ROI__STRUCT_H_
#define NAOQI_BRIDGE_MSGS__MSG__DETAIL__BODY_ROI__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/BodyROI in the package naoqi_bridge_msgs.
typedef struct naoqi_bridge_msgs__msg__BodyROI
{
  float angle;
  float cx;
  float cy;
  float height;
  float width;
  float confidence;
} naoqi_bridge_msgs__msg__BodyROI;

// Struct for a sequence of naoqi_bridge_msgs__msg__BodyROI.
typedef struct naoqi_bridge_msgs__msg__BodyROI__Sequence
{
  naoqi_bridge_msgs__msg__BodyROI * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} naoqi_bridge_msgs__msg__BodyROI__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // NAOQI_BRIDGE_MSGS__MSG__DETAIL__BODY_ROI__STRUCT_H_