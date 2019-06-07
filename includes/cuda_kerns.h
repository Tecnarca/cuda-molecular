#ifndef CUDA_KERNEL_HEAD
#define CUDA_KERNEL_HEAD

extern "C" void ps_kern(float* in, float* out, float precision, float* score_pos, int* start, int* stop, int* mask);

#endif

