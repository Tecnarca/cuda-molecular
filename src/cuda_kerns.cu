#ifndef DEBUG
#define INSIZE 256 //dimension of 'in'. Used items: 192 (64*3)
#define N_ATOMS 64 //atoms inside 'in'
#define N_FRAGS 4 //number of fragments
#define DEBUG 1 //should we execute the debug prints and checks?
#define MASKSIZE 256 //dimension of the 'mask'
#define VOLUMESIZE 1000000 //dimension of 'score_pos'
#define MAX_ANGLE 256 //up to which angle we need to run the algorithm?
#define LIMIT_DISTANCE2 2.0 //used in fragment_is_bumping, it is the minimum distance between to atoms
#define GRID_FACTOR_D 0.5
#define PI 3.141592653589793238462643383279
#define RADIAN_COEF PI/128.0;
#endif
#include <cuda_kerns.h>
#include <stdio.h>
#include <chrono>

texture<float, 1, cudaReadModeElementType> texScore_pos;
texture<int, 1, cudaReadModeElementType> texMask;

__inline__ __device__ int warpReduce(int val) {
	
	for (int i = warpSize/2; i > 0; i/=2){
		val += __shfl_down_sync(0xffffffff, val, i, 32);
	}
	
	return val;
}

__inline__ __device__ int blockReduce(int val) {
	
	static __shared__ int shared[32];

	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduce(val);

	if (lane==0) shared[wid]=val;
	
	__syncthreads();

	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
	
	if (wid==0) val = warpReduce(val);
	
	return val;
}

__device__ void compute_matrix( const int rotation_angle,
								const float x_orig, const float y_orig, const float z_orig,
								const float x_vector, const float y_vector, const float z_vector, float* matrix){

	const float u = (float)x_vector - x_orig;
	const float v = (float)y_vector - y_orig;
	const float w = (float)z_vector - z_orig;
	const float u2 = u * u;
	const float v2 = v * v;
	const float w2 = w * w;

	const float l2 = u * u + v * v + w * w;
	const float l = sqrtf(l2);

	const float angle_rad = (float)-rotation_angle*RADIAN_COEF;
	const float sint = sin(angle_rad);
	const float cost = cos(angle_rad);
	const float one_minus_cost = (float)1.0 - cost;

	matrix[0] =	(u2 + (v2 + w2) * cost) / l2;
	matrix[1] =	(u* v * one_minus_cost - w* l * sint) / l2;
	matrix[2] =	(u* w * one_minus_cost + v* l * sint) / l2;
	matrix[3] =	((x_orig * (v2 + w2) - u * (y_orig * v + z_orig * w)) * one_minus_cost + (y_orig * w - z_orig * v) * l * sint) / l2;

	matrix[4] =	(u* v * one_minus_cost + w* l * sint) / l2;
	matrix[5] =	(v2 + (u2 + w2) * cost) / l2;
	matrix[6] =	(v* w * one_minus_cost - u* l * sint) / l2;
	matrix[7] =	((y_orig * (u2 + w2) - v * (x_orig * u + z_orig * w)) * one_minus_cost + (z_orig * u - x_orig * w) * l * sint) / l2;

	matrix[8] =	(u* w * one_minus_cost - v* l * sint) / l2;
	matrix[9] =	(v* w * one_minus_cost + u* l * sint) / l2;
	matrix[10]=	(w2 + (u2 + v2) * cost) / l2;
	matrix[11]=	((z_orig * (u2 + v2) - w * (x_orig * u + y_orig * v)) * one_minus_cost + (x_orig * v - y_orig * u) * l * sint) / l2;
}

__global__ void rotate(float* in, int* mask, int iter, float precision, int* start, int* stop){
	
	const int index = blockIdx.x;
	const int curr_start = start[iter];
	const int curr_stop = stop[iter];
	const int x = threadIdx.x;
	const int y = threadIdx.x + N_ATOMS;
	const int z = threadIdx.x + 2*N_ATOMS;
	const int offset = ceil(index*INSIZE/precision);

	//This can be shared and computed only once per iter instance of rotate<<<,>>>() instead that once per thread!
	//but how to do it efficently? Shared memory? Then, how do i initialize it?
	float m[12];

	__shared__ float in_s[N_ATOMS*3];

	in_s[x] = in[x];
	in_s[y] = in[y];
	in_s[z] = in[z];

	__syncthreads();

	compute_matrix(index*precision,in_s[curr_start],in_s[curr_start+N_ATOMS],in_s[curr_start+2*N_ATOMS],in_s[curr_stop],in_s[curr_stop+N_ATOMS], in_s[curr_stop+2*N_ATOMS], m);

	//is this line correct? Can we optimize this access with a 2D texture of dimension (64,4)? (probably no)
	/*The line IS NOT correct! causes a memory error, detectable only by calling
	cudaMemoryTest() before the call of rotate. Not even cuda-memcheck catched it! We revert temporarly to standard non-texturized array...*/
	const int mask_x = mask[x+iter*N_ATOMS];/*tex1Dfetch(texMask, x+iter*N_ATOMS);*/

	if(mask_x == 1){
		in[x+offset] = m[0] * in_s[x] + m[1] * in_s[y] + m[2] * in_s[z] + m[3];
		in[y+offset] = m[4] * in_s[x] + m[5] * in_s[y] + m[6] * in_s[z] + m[7];
		in[z+offset] = m[8] * in_s[x] + m[9] * in_s[y] + m[10] * in_s[z] + m[11];
	} else {
		in[x+offset]=in_s[x];
		in[y+offset]=in_s[y];
		in[z+offset]=in_s[z];		
	}
}

__global__ void measure_shotgun (float* in, float* scores, int* shotgun, float precision, int iter){
	const int index = blockIdx.x;
	const int writers = threadIdx.x;
	const int x = threadIdx.x + index*INSIZE;
	const int y = threadIdx.x + index*INSIZE + N_ATOMS;
	const int z = threadIdx.x + index*INSIZE + 2*N_ATOMS;

	int index_x = (int) (in[x]*GRID_FACTOR_D);
	int index_y = (int) (in[y]*GRID_FACTOR_D);
	int index_z = (int) (in[z]*GRID_FACTOR_D);

	if(threadIdx.x==0) shotgun[index] = 0;
	__syncthreads();

	if (index_x < 0) index_x = 0;
	if (index_x > 100) index_x = 100;
	if (index_y < 0) index_y = 0;
	if (index_y > 100) index_y = 100;
	if (index_z < 0) index_z = 0;
	if (index_z > 100) index_z = 100;

	//Is this line correct? Can we optimize this access pattern with a 3D texture of dimension (100,100,100)? (probably yes)
	int score = scores[index_x+100*index_y+10000*index_z];/*tex1Dfetch(texScore_pos, index_x+100*index_y+10000*index_z);*/

	int reduced = blockReduce(score);
	if(!writers) shotgun[index] = reduced;
}

__global__ void fragment_is_bumping(float* in, int* mask, int* is_bumping_p, int iter, float precision){
	const int index = blockIdx.y;
	int ix = threadIdx.x;
	int jx = blockIdx.x;
	int iy = threadIdx.x + N_ATOMS;
	int jy = blockIdx.x + N_ATOMS;
	int iz = threadIdx.x + 2*N_ATOMS;
	int jz = blockIdx.x + 2*N_ATOMS;
	int offset = index*INSIZE;

	__shared__ float in_s[N_ATOMS*3];

	in_s[ix] = in[ix+offset];
	in_s[iy] = in[iy+offset];
	in_s[iz] = in[iz+offset];

	__syncthreads();

	const float diff_x = in_s[ix] - in_s[jx];
	const float diff_y = in_s[iy] - in_s[jy];
	const float diff_z = in_s[iz] - in_s[jz];
	const float distance2 = diff_x * diff_x +  diff_y * diff_y +  diff_z * diff_z;

	//Are these lines correct?
	int m_ix = mask[ix+iter*N_ATOMS];/*tex1Dfetch(texMask, ix+iter*N_ATOMS);*/
	int m_jx = mask[jx+iter*N_ATOMS];/*tex1Dfetch(texMask, xx+iter*N_ATOMS);*/

	int val_bit = (fabsf(m_ix - m_jx) == 1 && jx>ix && distance2 < LIMIT_DISTANCE2)? 1:0;

	int reduced = blockReduce(val_bit);
	if(!ix) is_bumping_p[jx+index*N_ATOMS] = reduced;
}

__global__ void fragment_reduce(int* is_bumping, int* is_bumping_p){
	const int index = blockIdx.x;
	int x = threadIdx.x;
	int val_bit = is_bumping_p[x+index*N_ATOMS];
	int reduced = blockReduce(val_bit);
	if(!x) is_bumping[index] = (reduced)? 1:0;
}


__inline__ __device__ void warpReduce(int ind, int sho, int bum, int &ret1, int &ret2, int &ret3) {
	int im, sm, bm;
	for (int i = warpSize/2; i > 0; i/=2){
		im = __shfl_down_sync(0xffffffff, ind, i, 32);
		sm = __shfl_down_sync(0xffffffff, sho, i, 32);
		bm = __shfl_down_sync(0xffffffff, bum, i, 32);
		if(!(bm > bum || (bum==bm && sho>=sm))){
			ind = im;
			sho = sm;
			bum = bm;
		}
	}
	ret1=ind;	
	ret2=sho;
	ret3=bum;
}

__inline__ __device__ int find_best(int* shotgun, int* bumping, int index){
	int shot = shotgun[index];
	int bum = bumping[index];
	int ind = index;
	static __shared__ int sharedI[32];
	static __shared__ int sharedS[32];
	static __shared__ int sharedB[32];

	int lane = index % warpSize;
	int wid = index / warpSize;

	warpReduce(index, shot, bum, ind, shot, bum);

	if (lane==0){
		sharedI[wid]=ind;
		sharedS[wid]=shot;
		sharedB[wid]=bum;
	}
	
	__syncthreads();

	if(index < blockDim.x / warpSize){
		ind = sharedI[lane];
		bum = sharedB[lane];
		shot = sharedS[lane];
	} else {
		ind = 0;
		bum = 1;
		shot = 0;
	}
	
	if (wid==0) warpReduce(ind, shot, bum, ind, shot, bum);
	
	return ind;
}

__global__ void eval_angles(float* in, int* shotgun, int* bumping){
	
	__shared__ int best_angle;

	const int index = threadIdx.x;

	int best_index = find_best(shotgun, bumping, index);

	if(index == 0) {
		//printf("best: (%d: %f, %d, %d)\n", best_index, in[best_index*INSIZE], shotgun[best_index], bumping[best_index]);
		best_angle = best_index;
	}
	
	__syncthreads();

	//this line works assuming INSIZE<=MAX_ANGLE/precision. How do we remove this assumption? (probably need a for to copy multiple values)
	if(index < INSIZE) in[index] = in[best_angle*INSIZE+index];
}


#define cudaSafeCall(call)  \
        do {\
            cudaError_t err = call;\
            if (cudaSuccess != err) \
            {\
                printf("CUDA error in %s(%s): %s",__FILE__,__LINE__,cudaGetErrorString(err));\
                exit(EXIT_FAILURE);\
            }\
        } while(0)

void cudaMemoryTest()
{
    const unsigned int N = 1048576;
    const unsigned int bytes = N * sizeof(int);
    int *h_a = (int*)malloc(bytes);
    int *d_a;
    cudaSafeCall(cudaMalloc((int**)&d_a, bytes));

    memset(h_a, 0, bytes);
    cudaSafeCall(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
}

void ps_kern(float* in, float* out, float precision, float* score_pos, int* start, int* stop, int* mask)
{
	float *d_in, *d_score_pos;

	int *d_start, *d_stop, *d_mask, *d_shotgun;

	int *d_bumping, *d_bumping_partial;

	cudaError_t status, status_cp, status_wb;
	cudaStream_t s1, s2;
	cudaEvent_t start_t, stop_t;

	status = cudaMalloc((void**) &d_in, sizeof(float)*INSIZE*ceil(MAX_ANGLE/precision));
	if(DEBUG && status!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status), __FILE__, __LINE__);

	status_cp = cudaMemcpy(d_in, in, sizeof(float)*INSIZE, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status_cp), __FILE__, __LINE__);

	status = cudaMalloc((void**) &d_start, sizeof(int)*N_ATOMS);
	if(DEBUG && status!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status), __FILE__, __LINE__);

	status_cp = cudaMemcpy(d_start, start, sizeof(int)*N_ATOMS, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status_cp), __FILE__, __LINE__);

	status = cudaMalloc((void**) &d_stop, sizeof(int)*N_ATOMS);
	if(DEBUG && status!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status), __FILE__, __LINE__);

	status_cp = cudaMemcpy(d_stop, stop, sizeof(int)*N_ATOMS, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status_cp), __FILE__, __LINE__);

	status = cudaMalloc((void**)&d_bumping, sizeof(int)*ceil(MAX_ANGLE/precision));
	if(DEBUG && status!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status), __FILE__, __LINE__);

	status = cudaMalloc((void**)&d_bumping_partial, sizeof(int)*ceil(MAX_ANGLE/precision)*N_ATOMS);
	if(DEBUG && status!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status), __FILE__, __LINE__);

	status = cudaMalloc((void**)&d_shotgun, sizeof(int)*ceil(MAX_ANGLE/precision));
	if(DEBUG && status!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status), __FILE__, __LINE__);

	status = cudaMalloc((void**) &d_mask, sizeof(int)*MASKSIZE);
	if(DEBUG && status!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status), __FILE__, __LINE__);

	status_cp = cudaMemcpy(d_mask, mask, sizeof(int)*MASKSIZE, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status_cp), __FILE__, __LINE__);

	status = cudaMalloc((void**) &d_score_pos, sizeof(float)*VOLUMESIZE);
	if(DEBUG && status!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status), __FILE__, __LINE__);

	status_cp = cudaMemcpy(d_score_pos, score_pos, sizeof(float)*VOLUMESIZE, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status_cp), __FILE__, __LINE__);

	cudaResourceDesc resDesc1;
	memset(&resDesc1, 0.0, sizeof(resDesc1));
	resDesc1.resType = cudaResourceTypeLinear;
	resDesc1.res.linear.devPtr = d_score_pos;
	resDesc1.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc1.res.linear.desc.x = 32;
	resDesc1.res.linear.sizeInBytes = VOLUMESIZE*sizeof(float);

	cudaResourceDesc resDesc2;
	memset(&resDesc2, 0, sizeof(resDesc2));
	resDesc2.resType = cudaResourceTypeLinear;
	resDesc2.res.linear.devPtr = d_mask;
	resDesc2.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc2.res.linear.desc.x = 32;
	resDesc2.res.linear.sizeInBytes = MASKSIZE*sizeof(int);

	cudaTextureDesc texDesc1;
	memset(&texDesc1, 0.0, sizeof(texDesc1));
	texDesc1.readMode = cudaReadModeElementType;

	cudaTextureDesc texDesc2;
	memset(&texDesc2, 0, sizeof(texDesc2));
	texDesc2.readMode = cudaReadModeElementType;

	cudaTextureObject_t texScore_pos=0;
	cudaTextureObject_t texMask=0;
	cudaCreateTextureObject(&texScore_pos, &resDesc1, &texDesc1, NULL);
	cudaCreateTextureObject(&texMask, &resDesc2, &texDesc2, NULL);

	cudaEventCreate(&start_t);
	cudaEventCreate(&stop_t);

	cudaStreamCreate(&s1);
	cudaStreamCreate(&s2);

	dim3 bump_blocks(N_ATOMS,ceil(MAX_ANGLE/precision));

	cudaEventRecord(start_t);

	/*cudaMemoryTest() calls and the function itself can be removed in the future, when we solved all the errors*/
	for (int i=0;i<N_FRAGS;++i){

		rotate<<<ceil(MAX_ANGLE/precision),N_ATOMS,0,s1>>>(d_in, d_mask, i, precision, d_start, d_stop);
		cudaMemoryTest();

		cudaStreamSynchronize(s1);
		cudaStreamSynchronize(s2);

		fragment_is_bumping<<<bump_blocks,N_ATOMS,0,s1>>>(d_in, d_mask, d_bumping_partial, i, precision);
		cudaMemoryTest();
		
		measure_shotgun<<<ceil(MAX_ANGLE/precision),N_ATOMS,0,s2>>>(d_in, d_score_pos, d_shotgun, precision, i);
		cudaMemoryTest();
		
		fragment_reduce<<<ceil(MAX_ANGLE/precision),N_ATOMS,0,s1>>>(d_bumping, d_bumping_partial);
		cudaMemoryTest();
		
		cudaStreamSynchronize(s1);
		cudaStreamSynchronize(s2);

		eval_angles<<<1,ceil(MAX_ANGLE/precision),0,s1>>>(d_in, d_shotgun, d_bumping);
		cudaMemoryTest();
	}

	cudaDeviceSynchronize();

	cudaEventRecord(stop_t);

	float milliseconds = 0;
	//Se chiamo la funzione qui sotto restituisce un errore cuda-memcheck: 
	//Program hit cudaErrorInvalidResourceHandle (error 400) due to "invalid resource handle" on CUDA API call to cudaEventElapsedTime.
	//Wtf? Forse è colpa degli stream?
	cudaEventElapsedTime(&milliseconds, start_t, stop_t);
	printf("\nKernels executed in %f milliseconds\n", milliseconds);

	status_wb = cudaMemcpy(out, d_in, sizeof(float)*INSIZE, cudaMemcpyDeviceToHost);
	if(DEBUG && status_wb!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status_wb), __FILE__, __LINE__);

	cudaDestroyTextureObject(texScore_pos);
	cudaDestroyTextureObject(texMask);
	cudaFree(d_bumping_partial);
	cudaEventDestroy(start_t);
	cudaEventDestroy(stop_t);
	cudaStreamDestroy(s1);
	cudaStreamDestroy(s2);
	cudaFree(d_bumping);
	cudaFree(d_shotgun);
	cudaFree(d_start);
	cudaFree(d_stop);
	cudaFree(d_in);
}
