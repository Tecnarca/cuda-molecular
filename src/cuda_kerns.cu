#ifndef DEBUG
#define INSIZE 192 //real (used by the main.cpp) dimension of 'in'
#define N_ATOMS 64 //atoms inside 'in'
#define N_FRAGS 4 //number of fragments
#define DEBUG 1 //should we execute the debug prints and checks?
#define MASKSIZE 256 //dimension of the 'mask'
#define VOLUMESIZE 1000000 //dimension of 'score_pos'
#define THREADSPERBLOCK 256 //how many threads per block? (max 1024) <- probabilmente c'è da vedere quanto conviene lasciarlo così
#define BLOCKS 1 //how many blocks per grid? <- probabilmente deve essere UNO
#define MAX_ANGLE 256 //up to which angle we need to run the algorithm?
#define LIMIT_DISTANCE2 2.0 //used in fragment_is_bumping, it is the minimum distance between to atoms
#define GRID_FACTOR_D 0.5
#define PI 3.141592653589793238462643383279
#define RADIAN_COEF PI/128.0;
#endif

#include <cuda_kerns.h>
#include <stdio.h>

texture<float> texScore_pos;
//di mask non stiamo a usà il fatto che è una texture (?)
texture<int> texMask;

__device__ void compute_matrix( const float rotation_angle,
                const float x_orig, const float y_orig, const float z_orig,
                const float x_vector, const float y_vector, const float z_vector, float* matrix){
            
            // compute the rotation axis
            const float u = (float)x_vector - x_orig;
            const float v = (float)y_vector - y_orig;
            const float w = (float)z_vector - z_orig;
            const float u2 = u * u;
            const float v2 = v * v;
            const float w2 = w * w;

            // compute its lenght and square root
            const float l2 = u * u + v * v + w * w;
            const float l = sqrtf(l2);

            // precompute sine and cosine for the angle
            float angle_rad = (float)-rotation_angle*RADIAN_COEF;
            const float sint = sin(angle_rad);
            const float cost = cos(angle_rad);
            const float one_minus_cost = (float)1.0 - cost;

            // return the actual matrix
            matrix[0] = 	   (u2 + (v2 + w2) * cost) / l2;
            matrix[1] =        (u* v * one_minus_cost - w* l * sint) / l2;
            matrix[2] =        (u* w * one_minus_cost + v* l * sint) / l2;
            matrix[3] =        ((x_orig * (v2 + w2) - u * (y_orig * v + z_orig * w)) * one_minus_cost + (y_orig * w - z_orig * v) * l * sint) / l2;

            matrix[4] =        (u* v * one_minus_cost + w* l * sint) / l2;
            matrix[5] =        (v2 + (u2 + w2) * cost) / l2;
            matrix[6] =        (v* w * one_minus_cost - u* l * sint) / l2;
            matrix[7] =        ((y_orig * (u2 + w2) - v * (x_orig * u + z_orig * w)) * one_minus_cost + (z_orig * u - x_orig * w) * l * sint) / l2;

            matrix[8] =        (u* w * one_minus_cost - v* l * sint) / l2;
            matrix[9] =        (v* w * one_minus_cost + u* l * sint) / l2;
            matrix[10] =        (w2 + (u2 + v2) * cost) / l2;
            matrix[11] =        ((z_orig * (u2 + v2) - w * (x_orig * u + y_orig * v)) * one_minus_cost + (x_orig * v - y_orig * u) * l * sint) / l2;
}

__global__ void rotate(float* in, int* mask, float angle, float* in_r, int iter, float precision, int curr_start, int curr_stop)
{
	if(angle == 0.0) {
		return;
	}
	
	// each thread will transform the coordinates of one atom
	int x = threadIdx.x;
	int y = threadIdx.x + blockDim.x;
	int z = threadIdx.x + 2*blockDim.x;
	float m[12];
	compute_matrix(angle*precision,in[curr_start],in[curr_start+N_ATOMS],in[curr_start+2*N_ATOMS],in[curr_stop],in[curr_stop+N_ATOMS], in[curr_stop+2*N_ATOMS], m);
	 
	int mask_x = tex1Dfetch(texMask, x+iter*N_ATOMS);
	 if(mask_x == 1){
		// gets current coordinates
		const float prev_x = in[x];
		const float prev_y = in[y];
		const float prev_z = in[z];
	    // update in (the one that is allocated in the GPU)
		//in_r[x] = m[0] * prev_x + m[1] * prev_y + m[2] * prev_z + m[3]; //268 Errori (di write): fare in_r[x] = 0; ne restituisce altrettanti
		//in_r[y] = m[4] * prev_x + m[5] * prev_y + m[6] * prev_z + m[7]; //204 Errori (di write): fare in_r[y] = 0; ne restituisce altrettanti
		//in_r[z] = m[8] * prev_x + m[9] * prev_y + m[10] * prev_z + m[11]; //268 Errori (di write): fare in_r[x] = 0; ne restituisce altrettanti
	}
}

//aggiustare in un blocco solo
__global__ void measure_shotgun (float* atoms, float* pocket, float* scores, int index)
{

    // one entry per atom processed within block (don't know if it's actually faster)
    __shared__ float cache[INSIZE];

    // each thread will process one atom
    int x = threadIdx.x;
    int y = threadIdx.x + blockDim.x;
    int z = threadIdx.x + 2*blockDim.x;

    unsigned int cacheIndex = x;

    int index_x = (int) (atoms[x]  * GRID_FACTOR_D);
    int index_y = (int) (atoms[y]  * GRID_FACTOR_D);
    int index_z = (int) (atoms[z]  * GRID_FACTOR_D);

    if (index_x < 0) index_x = 0;
    if (index_x > 100) index_x = 100;
    if (index_y < 0) index_y = 0;
    if (index_y > 100) index_y = 100;
    if (index_z < 0) index_z = 0;
    if (index_z > 100) index_z = 100;

    // perform reduction (compute partial block score)
    //cache[cacheIndex] = tex1Dfetch(texScore_pos, index_x+100*index_y+10000*index_z); //ERROR!

    __syncthreads();

    int i = blockDim.x/2;

    while (i != 0) {
      if (cacheIndex < i) {
        cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
      }
      i /= 2;
    }

    __syncthreads(); // probably unnecessary, or probably should be __threadfence_block()

    if(cacheIndex == 0) {
      if(index<0 || index > THREADSPERBLOCK-1)printf("(i: %d) ", index); //0 printfs
      scores[index] = cache[0];
      atoms[index] = 0;
    }
}


//THREAD NUMBER INCORRECT: INCAPSULATE
__global__ void fragment_is_bumping(float* in, const int* mask, bool* cache_is_bumping, int res){

	__shared__ bool all_bumps[INSIZE*(INSIZE-1)/2];
	//spawn n_atoms threads per block and n_atoms blocks
	int ix = threadIdx.x; 
	int jx = threadIdx.y; 
	int iy = threadIdx.x + blockDim.x; 
	int jy = threadIdx.y + blockDim.x; 
	int iz = threadIdx.x + 2*blockDim.x; 
	int jz = threadIdx.y + 2*blockDim.x;
	//Unique sequential index for threads with jx>ix
	int cacheIndex = ix*(INSIZE-1)-ix*(ix-1)+(ix-1)*ix/2+jx-ix-1;
	
	if(jx>ix){
		//int m_ix = tex1Dfetch(texMask, ix);
		//int m_jx = tex1Dfetch(texMask, jx);
		//if(fabsf(m_ix - m_jx) == 1){

			const float diff_x = in[ix] - in[jx];
	        const float diff_y = in[iy] - in[jy];
	        const float diff_z = in[iz] - in[jz];
	        
	        const float distance2 = diff_x * diff_x +  diff_y * diff_y +  diff_z * diff_z;

			if (distance2 < LIMIT_DISTANCE2) all_bumps[cacheIndex] = true;
			else all_bumps[cacheIndex] = false;

			__syncthreads();

    		int i = INSIZE*(INSIZE-1)/4;

    		while (i != 0) {
	      		if (cacheIndex < i) {
	        		all_bumps[cacheIndex] |= all_bumps[cacheIndex + i];
	        		__syncthreads();
	      		}
      			i /= 2;
    		}

    		if(cacheIndex == 0) {
      			//cache_is_bumping[res] = all_bumps[cacheIndex];
    		}
	}
}

__global__ void eval_angles(float* in, float* score_pos, int curr_start, int curr_stop, int* mask, float precision, int iter) {

  	// each thread will evaluate an angle and put the score on the cache (cache_score)
  	int angle = threadIdx.x, best_angle=0;

	// I want to get the angle that scores most without bumping, so I need 2 caches
	__shared__ float cache_score[THREADSPERBLOCK];
	__shared__ bool cache_is_bumping[THREADSPERBLOCK];

	// I found that this cache is necessary in the reduction phase to avoid WAWs
	__shared__ int best_angles[THREADSPERBLOCK];

	unsigned int cacheIndex = angle;
	


	dim3 bumpdim(INSIZE,INSIZE,1);
      
	//printf("%f",in[angle*N_ATOMS]);

	  best_angles[cacheIndex] = angle;

	  cache_is_bumping[angle]=0;

	  rotate<<<BLOCKS,THREADSPERBLOCK>>>(in, mask, angle, &in[angle*N_ATOMS], iter, precision, curr_start, curr_stop);

	  measure_shotgun<<<1, INSIZE >>>(&in[angle*N_ATOMS], score_pos, cache_score, angle);  // populates the scores cache

	  //if(DEBUG) printf("score is: %d for fragm %d with angle %f\n", cache_score, angle, angle*precision);
	  
	  //fragment_is_bumping<<<1,bumpdim>>>(&in[angle*N_ATOMS], mask, cache_is_bumping, angle); // populates the is_bumping cache

		// doubt: I don't know if I can pass shared caches to other nested kernels in total tranquility (I think so).
		// If not, we need a "buffer" array for the function that will be flushed into the cache (not so bad)

	  unsigned int i = blockDim.x/2;

	  // get the highest non-bumping score
	  while (i != 0) {
	    if (cacheIndex < i) {
	      if (cache_score[cacheIndex] < cache_score[cacheIndex + i] && !cache_is_bumping[cacheIndex + i]) {
	        cache_score[cacheIndex] = cache_score[cacheIndex + i];
	        best_angles[cacheIndex] = cacheIndex + i;
	      }
	      __syncthreads();
	    }
	    i /= 2;
	  }

	  if(cacheIndex == 0) {
	    //best_score = cache_score[cacheIndex];
	    best_angle = best_angles[cacheIndex];
	  	rotate<<<BLOCKS,THREADSPERBLOCK>>>(in, mask, best_angle, in, iter, precision, curr_start, curr_stop);
	  }
}

//Speed improvements:
//0) Implement CONSTANT and TEXTURE memory effectively
//1) (having n_atoms as global) remove the indexing by "+blockDim.x" in the kernels to use "+N_ATOMS", that is a predefined variable (so it's faster to access?)
//2) If we want to have a greater parallelism, we could put the malloc and the memcpy on different streams (not requested for this project by the professor)
//3) Check if it's convenient to remove the 'i' parameter from some kernels and use the '&mask[i*N_ATOMS]' instead
//4) METTERE GLI STREAM

void ps_kern(float* in, float* out, float precision, float* score_pos, int* start, int* stop, int* mask, int n_frags)
{
	float *d_in, *d_score_pos, d_best_angle;
	int *d_mask, d_best_score;

	cudaError_t status, status_cp, status_tx, status_wb;;

	//GPU MEMORY INITIALIZATION
	
	status = cudaMalloc((void**) &d_in, sizeof(float)*INSIZE*MAX_ANGLE);
	if(DEBUG && status!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status), __FILE__, __LINE__);

	//status_cp = cudaMemcpy(d_in, in, sizeof(float)*INSIZE, cudaMemcpyHostToDevice);
	cudaMemset(d_in, 0.0, sizeof(float)*INSIZE*MAX_ANGLE);
	if(DEBUG && status_cp!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status_cp), __FILE__, __LINE__);

	status = cudaMalloc((void**) &d_score_pos, sizeof(float)*VOLUMESIZE);
	if(DEBUG && status!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status), __FILE__, __LINE__);

	status_cp = cudaMemcpy(d_score_pos, score_pos, sizeof(float)*VOLUMESIZE, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status_cp), __FILE__, __LINE__);

	status = cudaMalloc((void**) &d_mask, sizeof(int)*MASKSIZE);
    if(DEBUG && status!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status), __FILE__, __LINE__);

	status_cp = cudaMemcpy(d_mask, mask, sizeof(int)*MASKSIZE, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status_cp), __FILE__, __LINE__);

	status = cudaMalloc((void**) &d_best_score, sizeof(int));
    if(DEBUG && status!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status), __FILE__, __LINE__);

	status = cudaMalloc((void**) &d_best_angle, sizeof(float));
    if(DEBUG && status!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status), __FILE__, __LINE__);

  // --------------
	status_tx = cudaBindTexture(NULL, texScore_pos, d_score_pos, sizeof(float)*VOLUMESIZE);
	if(DEBUG && status_tx!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status_tx), __FILE__, __LINE__);

	status_tx = cudaBindTexture(NULL, texMask, d_mask, sizeof(int)*MASKSIZE);
	if(DEBUG && status_tx!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status_tx), __FILE__, __LINE__);				

	// start CUDA timing here
	for(int i=0; i<N_FRAGS; i++){
		//if(DEBUG) printf("ITER: %d\n", i);
		eval_angles<<<BLOCKS,MAX_ANGLE>>>(d_in, d_score_pos, start[i], stop[i], d_mask, precision, i);
		//if(DEBUG) cudaDeviceSynchronize();
	}
	cudaDeviceSynchronize();
	//stop CUDA timing here

	status_wb = cudaMemcpy(out, d_in, sizeof(float)*INSIZE, cudaMemcpyDeviceToHost);
	if(DEBUG && status_wb!=cudaSuccess)
		printf("%s in %s at line %d\n", cudaGetErrorString(status_wb), __FILE__, __LINE__);
	
	cudaUnbindTexture(texScore_pos);
	cudaUnbindTexture(texMask);
	cudaFree(d_score_pos);
	cudaFree(d_mask);
	cudaFree(d_in);
}
