cmake_minimum_required(VERSION 3.15...3.20)

project(
  Armin
  VERSION "1.0.1"
  DESCRIPTION "Armin++ Project"
  LANGUAGES CUDA C CXX)

include(FetchContent)
include(CheckCXXCompilerFlag)
include(CMakePrintHelpers)

message("DIR THRUST : " ${Thrust_DIR})
find_package(Thrust REQUIRED CONFIG PATHS "/home/remi/lib/thrust/thrust/cmake")
thrust_create_target(Thrust)

add_executable(simulation simulation.cu)
target_link_libraries(simulation Thrust)