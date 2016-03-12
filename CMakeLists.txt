cmake_minimum_required(VERSION 3.3)
project(cs4096)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp neural_network.cpp neural_network.h)
add_executable(cs4096 ${SOURCE_FILES})