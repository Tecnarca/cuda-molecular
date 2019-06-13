project(HELLO_CUDA LANGUAGES CUDA CXX)

# CMakeLists.txt to build 
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
set (CMAKE_CXX_STANDARD 11)

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_52,code=sm_52 -std=c++11")

include_directories(
	includes
)

cuda_add_executable(
    hellocuda
    src/cuda_kerns.cu
    src/main.cpp
)
