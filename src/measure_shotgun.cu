// I think pocket should be texturizable

__device__ int measure_shotgun (float* atoms, float* pocket)
	{
    unsigned int threadsPerBlock = blockDim.x;

    // one entry per atom processed within block (don't know if it's actually faster)
    __shared__ float cache[threadsPerBlock];

    // each thread will process one atom
    int x = threadId.x;
		int y = threadId.x + blockDim.x;
		int z = threadId.x + 2*blockDim.x;

    unsigned int cacheIndex = x;
    float blockScore[gridDim.x] // one entry per block

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
    cache[cacheIndex] = pocket[index_x+100*index_y+10000*index_z];

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

    __syncthreads(); // probably unnecessary

    // total score
    return blockScore[0] + blockScore[1] + blockScore[2];
  }





		// loop over the atoms and get the pocket score
		for ( int i =0; i < n_atoms; ++i )
		{
			// compute the index inside of the pocket
			int index_x = static_cast<int>(atoms[i]  * grid_factor_d );
			int index_y = static_cast<int>(atoms[i+n_atoms]  * grid_factor_d );
			int index_z = static_cast<int>(atoms[i+2*n_atoms]  * grid_factor_d );
			if (index_x < 0) index_x = 0;
			if (index_x > 100) index_x = 100;
			if (index_y < 0) index_y = 0;
			if (index_y > 100) index_y = 100;
			if (index_z < 0) index_z = 0;
			if (index_z > 100) index_z = 100;
			// update the score value
			score += pocket[index_x+100*index_y+10000*index_z];
		}
		return score;
	}
