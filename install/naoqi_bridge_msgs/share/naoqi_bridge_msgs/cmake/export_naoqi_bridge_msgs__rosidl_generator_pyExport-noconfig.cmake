#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "naoqi_bridge_msgs::naoqi_bridge_msgs__rosidl_generator_py" for configuration ""
set_property(TARGET naoqi_bridge_msgs::naoqi_bridge_msgs__rosidl_generator_py APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(naoqi_bridge_msgs::naoqi_bridge_msgs__rosidl_generator_py PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libnaoqi_bridge_msgs__rosidl_generator_py.so"
  IMPORTED_SONAME_NOCONFIG "libnaoqi_bridge_msgs__rosidl_generator_py.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS naoqi_bridge_msgs::naoqi_bridge_msgs__rosidl_generator_py )
list(APPEND _IMPORT_CHECK_FILES_FOR_naoqi_bridge_msgs::naoqi_bridge_msgs__rosidl_generator_py "${_IMPORT_PREFIX}/lib/libnaoqi_bridge_msgs__rosidl_generator_py.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)