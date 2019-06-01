#ifndef DEBUG
#define INSIZE 256 //real (used by the main.cpp) dimension of 'in'
#define N_ATOMS 64 //atoms inside 'in'
#define N_FRAGS 4 //number of fragments
#define DEBUG 1 //should we execute the debug prints and checks?
#define MASKSIZE 256 //dimension of the 'mask'
#define VOLUMESIZE 1000000 //dimension of 'score_pos'
#define THREADSPERBLOCK 256 //how many threads per block? (max 1024) <- probabilmente c'è da vedere quanto conviene lasciarlo così
#define BLOCKS 3 //how many blocks per grid? <- probabilmente c'è da vedere quanto conviene lasciarlo così
#define MAX_ANGLE 256 //up to which angle we need to run the algorithm?
#define LIMIT_DISTANCE2 = 2.0f; //used in fragment_is_bumping, it is the minimum distance between to atoms
#endif

#include <iostream>
__device__ void rotate( float* in, float* atoms, float* rotation_matrix )
{
	// each thread will transform the coordinates of one atom
	int x = threadId.x;
	int y = threadId.x + blockDim.x;
	int z = threadId.x + 2*blockDim.x;

        // gets current coordinates
	const float prev_x = in[x];
	const float prev_y = in[y];
	const float prev_z = in[z];

        // one read is better than three (rotation_matrix is on constant memory)
	const float m11 = rotation_matrix[0];
	const float m12 = rotation_matrix[1];
	const float m13 = rotation_matrix[2];
	const float m14 = rotation_matrix[3];
	const float m21 = rotation_matrix[4];
	const float m22 = rotation_matrix[5];
	const float m23 = rotation_matrix[6];
	const float m24 = rotation_matrix[7];
	const float m31 = rotation_matrix[8];
	const float m32 = rotation_matrix[9];
    const float m33 = rotation_matrix[10];
	const float m34 = rotation_matrix[11];

    // update in (the one that is allocated in the GPU)
	in[x] = m11 * prev_x + m12 * prev_y + m13 * prev_z + m14;
	in[y] = m21 * prev_x + m22 * prev_y + m23 * prev_z + m24;
	in[z] = m31 * prev_x + m32 * prev_y + m33 * prev_z + m34;

}


__global__ void measure_shotgun (float* atoms, float* pocket, float* scores, int index)
{
    unsigned int threadsPerBlock = THREADSPERBLOCK;
    unsigned int blocks = BLOCKS;

    // one entry per atom processed within block (don't know if it's actually faster)
    __shared__ float cache[threadsPerBlock];

    // each thread will process one atom
    int x = threadId.x;
    int y = threadId.x + blockDim.x;
    int z = threadId.x + 2*blockDim.x;

    unsigned int cacheIndex = x;
    float blockScore[blocks] // one entry per block

    // get the average score
    int score = 0;

    int index_x = atoms[x]  * grid_factor_d;
    int index_y = atoms[y]  * grid_factor_d;
    int index_z = atoms[z]  * grid_factor_d;

    if (index_x < 0) index_x = 0;
    if (index_x > 100) index_x = 100;
    if (index_y < 0) index_y = 0;
    if (index_y > 100) index_y = 100;
    if (index_z < 0) index_z = 0;
    if (index_z > 100) index_z = 100;

    // perform reduction (compute partial block score)
    cache[cacheIndex] = tex1Dfetch(pocket, index_x+100*index_y+10000*index_z);

    __syncthreads();

    int i = blockDim.x/2;

    while (i != 0) {
      if (cacheIndex < i) {
        cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
      }
      i /= 2;
    }

    if(cacheIndex == 0) {
      blockScore[BlockId.x] = cache[cacheIndex];
    }

    __syncthreads(); // probably unnecessary, or probably should be __threadfence_block()

    // total score
    scores[index] = blockScore[0] + blockScore[1] + blockScore[2];
}


//I pass is_bumping for thread from each block to access it
__device__ void fragment_is_bumping(const float* in, const int* mask, bool &cache_is_bumping, int i){

	//trovare il modo di mettere is_bumping a 0
	__shared__ int is_bumping=0;

	//spawn n_atoms threads per block and n_atoms blocks
	const int ix = threadId.x; 
	const int jx = BlockId.x; 
	const int iy = threadId.x + blockDim.x; 
	const int jy = BlockId.x + blockDim.x; 
	const int iz = threadId.x + 2*blockDim.x; 
	const int jz = BlockId.x + 2*blockDim.x;

	if(jx>ix){
		if(fabsf(mask[ix]-mask[jx]) == 1){

			const float diff_x = in[ix] - in[jx];
	        const float diff_y = in[iy] - in[jy];
	        const float diff_z = in[iz] - in[jz];
	        
	        const float distance2 = diff_x * diff_x +  diff_y * diff_y +  diff_z * diff_z;
	        
	        //there could be multiple accesses to is_bumping! 
	        //Should we avoid it? (WAW to same value, there should be no problem)
			if (distance2 < LIMIT_DISTANCE2) is_bumping = 1;
		}
	}

	//provare a usare il log2()
	cache_is_bumping[i] = is_bumping;


}


__global__ void eval_angles(float* in, float* score_pos, int& best_angle, int& best_score, free_rotation::value_type& best_rotation_matrix, int start_index, int stop_index, int* mask) {

	//da controllare ;a dim dei thread
  int threadsPerBlock = THREADSPERBLOCK;
  int Blocks = BLOCKS;
  int cacheDim = blockDim.x;

  // each thread will evaluate an angle and put the score on the cache (cache_score)
  //Modifica: passa precision
  int angle = threadId.x;

  //precision = 0.6
  //threadId = 1 -> angolo = precision
  //threadId = 2 -> angolo = 2*precision

  	//implementa compute_matrix e trasforma in float l'angolo
	compute_matrix(angle,in[start_index],in[start_index+NATOMS],in[start_index+2*NATOMS],in[stop_index],in[stop_index+NATOMS], in[stop_index+2*NATOMS]);
	rotate<<<threadsPerBlock,Blocks>>>(in, &mask[i*n_atoms], rotation_matrix);

  // I want to get the angle that scores most without bumping, so I need 2 caches
  __shared__ float cache_score[cacheDim];
  __shared__ bool cache_is_bumping[cacheDim];

  // I found that this cache is necessary in the reduction phase to avoid WAWs
  __shared__ int best_angles[threadsPerBlock];
  unsigned int cacheIndex = angle;
  best_angles[cacheIndex] = angle;

  //ricontrollare conversione index <-> angle
  measure_shotgun<<<threadsPerBlock/Blocks,Blocks>>>(in, score_pos, &cache_score, angle);  // populates the scores cache
  
  if(DEBUG) printf("score is: %d for fragm %d with angle %.4f\n", score, i, j);
  
  fragment_is_bumping<<<threadsPerBlock,256>>>(in, &mask[i*n_atoms], &cache_is_bumping, angle); // populates the is_bumping cache

	// doubt: I don't know if I can pass shared caches to other nested kernels in total tranquility (I think so).
	// If not, we need a "buffer" array for the function that will be flushed into the cache (not so bad)

  __syncthreads();

  unsigned int i = blockDim.x/2;

  // get the highest non-bumping score
  while (i != 0) {
    if (cacheIndex < i) {
      if (cache_score[cacheIndex] < cache_score[cacheIndex + i] && !cache_is_bumping[cacheIndex + i]) {
        cache[cacheIndex] = cache[cacheIndex + i];
        best_angle[cacheIndex] = cacheIndex + i;
      }
      __syncthreads();
    }
    i /= 2;
  }

  if(cacheIndex == 0) {
    best_score = cache[cacheIndex];
    best_angle = best_angle[cacheIndex];
  	//ricalcolo la matrice
  	//modifico in
  }
}

//Speed improvements:
//0) Implement CONSTANT and TEXTURE memory effectively
//1) (having n_atoms as global) remove the indexing by "+blockDim.x" in the kernels to use "+N_ATOMS", that is a predefined variable (so it's faster to access?)
//2) If we want to have a greater parallelism, we could put the malloc and the memcpy on different streams (not requested for this project by the professor)
//3) Check if it's convenient to remove the 'i' parameter from some kernels and use the '&mask[i*N_ATOMS]' instead
//4) [put other ideas to test later here]

void ps_kern(float* in, float* out, int precision, float* score_pos, int* start, int* stop, int* mask )
{
	float *d_in, *d_score_pos, *d_rotation_matrix;
	int *d_mask;

	cudaError_t status, status_cp, status_tx;
	
	//Dimensions of block and grid for the 'fragment_is_bumping' functions
	dim3 bumping_block(INSIZE,1,1); //256,1,1
	dim3 bumping_grid(INSIZE,MAX_ANGLE*1/precision,1); //256,256,1

	//GPU MEMORY INITIALIZATION
	texture<float> texScore_pos;
	texture<int> texMask;
	
	status = cudaMalloc((void**) &d_in, sizeof(float)*INSIZE);
	if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status_cp = cudaMemcpy(d_in, in, sizeof(float)*INSIZE, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		cout << cudaGetErrorString(status_cp) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status = cudaMalloc((void**) &d_score_pos, sizeof(float)*VOLUMESIZE);
	if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status_cp = cudaMemcpy(d_score_pos, score_pos, sizeof(float)*VOLUMESIZE, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		cout << cudaGetErrorString(status_cp) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status = cudaMalloc((void**) &d_mask, sizeof(int)*MASKSIZE);
  if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status_cp = cudaMemcpy(d_mask, mask, sizeof(int)*MASKSIZE, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		cout << cudaGetErrorString(status_cp) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	//Da controllare
	status = cudaMalloc((void**) &d_rotation_matrix, sizeof(float)*12);
  if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status_cp = cudaMemcpy(d_rotation_matrix, rotation_matrix, sizeof(float)*12, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		cout << cudaGetErrorString(status_cp) << " in " << __FILE__ << " at line " << __LINE__ << endl;	

  // --------------
	status_tx = cudaBindtexture(NULL, texScore_pos, d_score_pos, sizeof(float)*VOLUMESIZE);
	if(DEBUG && status_tx!=cudaSuccess)
	  cout << cudaGetErrorString(status_tx) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status_tx = cudaBindtexture(NULL, texMask, d_mask, sizeof(int)*MASKSIZE);
	if(DEBUG && status_tx!=cudaSuccess)
		 cout << cudaGetErrorString(status_tx) << " in " << __FILE__ << " at line " << __LINE__ << endl;				


	//CUDA stream creation
	//https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf
	//The webminar above explains how to use effectively the cuda streams
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	// start CUDA timing here

	for (int i=0;i<n_frags;++i){ //Rotameter optimization. Numbers after the cells is the stream channel.

		// get the index of starting atom
		const auto start_atom_index = start[i];
		const auto stop_atom_index = stop[i];

		int best_angle = 0;
		float best_score = 0.0;
		
		free_rotation::value_type best_rotation_matrix; // Da controllare

		//Da controllare dimensioni dei kernel
		eval_angles<<<256*ceil(1/precision),1>>>(in, score_pos, best_angle, best_score, best_rotation_matrix, start_atom_index, stop_atom_index, mask);
		
		//get matrice
		//calcolare il nuovo in
		//memcopy in gpu

		std::cout<<"best angle is: "<<best_angle<<std::endl;
		std::cout<<" score is: "<<best_score<<" for fragm: "<<i<<"with angle out"<<std::endl;
	}
	//stop CUDA timing here
		
	cudaError_t status_wb;
	
	//we (should) copy the best 'in' array back in central memory as the 'out' array 
	//(this memcpy copies from the gpu the FIRST 'in' array (index 0-255) that isn't necessarily the best, unless "compute_best" ensures this)
	
	//dove finisce l'output?
	status_wb = cudaMemcpy(out, d_in, sizeof(float)*ARRSIZE, cudaMemcpyDeviceToHost);
	if(DEBUG && status_wb!=cudaSuccess)
		cout << cudaGetErrorString(status_wb) << " in " << __FILE__ << " at line " << __LINE__ << endl;
	
	cudaUnbindTexture(texScore_pos);
	cudaUnbindTexture(texMask);
	cudaFree(d_score_pos);
	cudaFree(d_mask);
	cudaFree(d_rotation_matrix);
	cudaFree(d_in);
}


