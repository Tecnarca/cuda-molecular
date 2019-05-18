#ifndef CUDA_KERNEL_HEAD
#define CUDA_KERNEL_HEAD

//extern "C" void align_kern(float* atoms_in, float* atoms_out, int precision,float* score_pos );
extern "C" void ps_kern(float* atoms_in, float* atoms_out, int precision,float* score_pos, int* start, int* stop, int* mask );
//extern "C" void vibro_kern(float* atoms_in, float* atoms_out, int precision,int* score_pos );

#endif

