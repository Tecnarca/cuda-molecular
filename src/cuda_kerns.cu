#ifndef ARRSIZE
#define ARRSIZE 256*512*512
#endif

#ifndef DEBUG
#define DEBUG 1
#define NATOM 4
#define MASKSIZE 256
#define VOLUMESIZE 1000000
#endif

<<<<<<< HEAD
#define THREADSPERBLOCK 512
#define BLOCKS 3
=======
// __constant__ float rotation_matrix[12]

>>>>>>> 9171ba49fae4808d3f9c256c218325df3b6de2d9

// function declarations (see bodies below). We may create an header file for these ones, but then we have to modify CMakeLists.txt I guess
__global__ void fragment_check(float*, float*, int&, int&, int*, int*, int*);
__global__ void update(float*, float*);

extern "C" void ps_kern( float* in, float* out, int precision, float* score_pos, int* start, int* stop, int* mask )
{
	float *d_in, *d_out, *d_score_pos;

	int* d_start, int* d_stop, int* d_mask;

	cudaError_t status;
	cudaError_t status_cp;

	status = cudaMalloc((void**) &d_in, sizeof(float)*ARRSIZE);
	if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status_cp = cudaMemcpy(d_in, in, sizeof(float)*ARRSIZE, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		cout << cudaGetErrorString(status_cp) << " in " << __FILE__ << " at line " << __LINE__ << endl;


	status = cudaMalloc((void**) &d_out, sizeof(float)*ARRSIZE);
	if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status_cp = cudaMemcpy(d_out, out, sizeof(float)*ARRSIZE, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		cout << cudaGetErrorString(status_cp) << " in " << __FILE__ << " at line " << __LINE__ << endl;


	status = cudaMalloc((void**) &d_score_pos, sizeof(float)*VOLUMESIZE);
	if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status_cp = cudaMemcpy(d_score_pos, score_pos, sizeof(float)*VOLUMESIZE, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		cout << cudaGetErrorString(status_cp) << " in " << __FILE__ << " at line " << __LINE__ << endl;


	status = cudaMalloc((void**) &d_start, sizeof(float)*NATOM);
	if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status_cp = cudaMemcpy(d_start, start, sizeof(float)*NATOM, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		cout << cudaGetErrorString(status_cp) << " in " << __FILE__ << " at line " << __LINE__ << endl;


	status = cudaMalloc((void**) &d_stop, sizeof(float)*NATOM);
  if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status_cp = cudaMemcpy(d_stop, stop, sizeof(float)*NATOM, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		cout << cudaGetErrorString(status_cp) << " in " << __FILE__ << " at line " << __LINE__ << endl;


	status = cudaMalloc((void**) &d_mask, sizeof(float)*MASKSIZE);
  if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status_cp = cudaMemcpy(d_mask, mask, sizeof(float)*MASKSIZE, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		cout << cudaGetErrorString(status_cp) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	// optimize each rotamer
	for ( int i = 0; i < n_frags; ++i )
	{
		// compute the epsilon value for floating point precision
		const auto epsilon = std::numeric_limits<float>::epsilon();

		// get the index of starting atom
		const auto start_atom_index = start[i];
		const auto stop_atom_index = stop[i];

		// get the rotation matrix for this fragment
		const auto rotation_matrix = free_rotation::compute_matrix(precision,in[start_atom_index],in[start_atom_index+n_atoms],in[start_atom_index+2*n_atoms],in[stop_atom_index],in[stop_atom_index+n_atoms],  in[stop_atom_index+2*n_atoms]);
		// declare the variables for driving the optimization
		int best_angle = 0;
		int best_score = measure_shotgun<<<THREADSPERBLOCK, BLOCKS>>>(d_in, d_score_pos);
		std::cout<<"init: best score is: "<<best_score<<std::endl;
		bool is_best_bumping = fragment_is_bumping<<<THREADSPERBLOCK, BLOCKS>>>(d_in, &d_mask[i*n_atoms]); // check the manipulations made on d_mask!
		// optimize shape

		fragment_check<<<256*(1/precision), 1>>>(best_angle, best_score); // Until the precision is not >1024 only 1 block needed

		std::cout<<"best angle is: "<<best_angle<<std::endl;

		const int score = measure_shotgun<<<threadsPerBlock,Blocks>>>(d_in, d_score_pos);
		std::cout<<" score is: "<<score<<" for fragm: "<<i<<"with angle out"<<std::endl;

		const auto rotation_matrix_best = free_rotation::compute_matrix(best_angle,in[start_atom_index],in[start_atom_index+n_atoms],in[start_atom_index+2*n_atoms],in[stop_atom_index],in[stop_atom_index+n_atoms],  in[stop_atom_index+2*n_atoms]);
		rotate<<<THREADSPERBLOCK, BLOCKS>>>(d_in, &d_mask[i*n_atoms], rotation_matrix_best);
	}

	update<<<THREADSPERBLOCK, BLOCKS>>>(d_in, d_out);

	cudaError_t status_wb;
	status_wb = cudaMemcpy(in, d_in, sizeof(float)*ARRSIZE, cudaMemcpyDeviceToHost);
	if(DEBUG && status_wb!=cudaSuccess)
		cout << cudaGetErrorString(status_wb) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status_wb = cudaMemcpy(out, d_out, sizeof(float)*ARRSIZE, cudaMemcpyDeviceToHost);
	if(DEBUG && status_wb!=cudaSuccess)
		cout << cudaGetErrorString(status_wb) << " in " << __FILE__ << " at line " << __LINE__ << endl;


	// none of these data structures is changed by any kernel, thus I think we can texturize
	cudaFree(d_score_pos);
	cudaFree(d_start);
	cudaFree(d_stop);

  // not texturizable
	cudaFree(d_in);
	cudaFree(d_out);

	// for this I don't know, depends on fragment_is_bumping kernel
	cudaFree(d_mask);

}




__global__ void fragment_check(float* in, float* score_pos, int& best_angle, int& best_score, int* start, int* stop, int* mask) {
  // precision is not passed because it determines the n. of threads

  int threadsPerBlock = blockDim.x;
  int Blocks = gridDim.x;  // don't know if it is correct

  // each thread will evaluate an angle and put the score on the cache (cache_score)
  int angle = threadId.x;

  // I want to get the angle that scores most without bumping, so I need 2 caches
  __shared__ float cache_score[threadsPerBlock];
  __shared__ bool cache_is_bumping[threadsPerBlock];

  // I found that this cache is necessary in the reduction phase, otherwise deadlocks are possible
  __shared__ int best_angles[threadsPerBlock];
  unsigned int cacheIndex = angle;
  best_angles[cacheIndex] = angle;

  const int score = measure_shotgun<<<threadsPerBlock,Blocks>>>(in, score_pos);
  std::cout<<" score is: "<<score<<" for fragm: "<<i<<"with angle: "<<j<<std::endl;

  cache_score[cacheIndex] = score;

  const bool is_bumping = fragment_is_bumping<<<threadsPerBlock,Blocks>>>(in, &mask[i*n_atoms]);

  cache_is_bumping[cacheIndex] = is_bumping;

  __syncthreads();

  int i = blockDim.x/2;

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
  }

  // iterate to next angle (for this fragment)
  rotate<<<threadsPerBlock,Blocks>>>(in, &mask[i*n_atoms], rotation_matrix);
}




__global__ void update(float* in, float* out) {

  int x = threadId.x;
  out[x] = in [x];
}
