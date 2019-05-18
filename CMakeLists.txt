# CMakeLists.txt to build 
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)



set (CMAKE_CXX_STANDARD 11)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
#    -O3 -gencode arch=compute_22,code=sm_22
    -std=c++11 -g
)

 
include_directories(
	includes
)



# Specify binary name and source file to build it from
cuda_add_executable(
    hellocuda
    src/cuda_kerns.cu
    src/main.cpp
#aca_proj.cu
)
