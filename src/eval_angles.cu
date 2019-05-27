__global__ void eval_angles(float* in, float* score_pos, int& best_angle, int& best_score, int start_index, int stop_index, int* mask) {

  int threadsPerBlock = THREADSPERBLOCK;
  int Blocks = BLOCKS;

  // each thread will evaluate an angle and put the score on the cache (cache_score)
  const float angle = threadId.x;

	const auto rotation_matrix = free_rotation::compute_matrix(angle,in[start_index],in[start_index+NATOMS],in[start_index+2*NATOMS],in[stop_index],in[stop_index+NATOMS], in[stop_index+2*NATOMS]);
	rotate<<<threadsPerBlock,Blocks>>>(in, &mask[i*n_atoms], rotation_matrix);

  // I want to get the angle that scores most without bumping, so I need 2 caches
  __shared__ float cache_score[threadsPerBlock];
  __shared__ bool cache_is_bumping[threadsPerBlock];

  // I found that this cache is necessary in the reduction phase to avoid WAWs
  __shared__ int best_angles[threadsPerBlock];
  unsigned int cacheIndex = angle;
  best_angles[cacheIndex] = angle;

  measure_shotgun<<<threadsPerBlock,Blocks>>>(in, score_pos, &cache_score);  // populates the scores cache
  std::cout<<" score is: "<<score<<" for fragm: "<<i<<"with angle: "<<j<<std::endl;  // probably problematic :)

  fragment_is_bumping<<<threadsPerBlock,Blocks>>>(in, &mask[i*n_atoms], &cache_is_bumping); // populates the is_bumping cache

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
  }
}
