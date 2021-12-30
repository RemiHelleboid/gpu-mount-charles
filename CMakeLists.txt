cmake_minimum_required(VERSION 3.15...3.20)

project(
  Armin
  VERSION "1.0.1"
  DESCRIPTION "Armin++ Project"
  LANGUAGES CUDA C CXX)

set(default_build_type "Release")
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(default_build_type "RelWithDebInfo")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(
    STATUS
      "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE
      "${default_build_type}"
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()

include(FetchContent)
include(CheckCXXCompilerFlag)
include(CMakePrintHelpers)

message("DIR THRUST : " ${Thrust_DIR})
find_package(Thrust REQUIRED CONFIG PATHS "/home/remi/lib/thrust/thrust/cmake")
thrust_create_target(Thrust)

add_executable(simulation simulation.cu)
target_link_libraries(simulation Thrust)
