project(HELLO_CUDA LANGUAGES CUDA CXX)

# CMakeLists.txt to build 
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA QUIET REQUIRED)
set (CMAKE_CXX_STANDARD 11)

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_52,code=sm_52 -rdc=true -std=c++11 -lcudadevrt -lcudart")



set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -arch=sm_52 -gencode arch=compute_52,code=sm_52 -rdc=true -std=c++11 -lcudadevrt -lcudart
)
 
include_directories(
	includes
	/usr/local/cuda-10.1/targets/x86_64-linux/lib
)

# Specify binary name and source file to build it from
cuda_add_executable(
    hellocuda
    src/cuda_kerns.cu
    src/main.cpp
#aca_proj.cu
)
