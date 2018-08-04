cmake_minimum_required(VERSION 3.10)
project(CNN_c)

SET(CMAKE_CXX_FLAGS "-g -Wall -std=c++14 -O2")
add_library(CNN_c SHARED library.cpp)
