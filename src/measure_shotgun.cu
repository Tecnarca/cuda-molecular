// I think pocket should be texturizable, as here it is only read contiguously by contiguous threads.

__global__ void measure_shotgun (float* atoms, float* pocket, float* scores, int index)
{
    unsigned int threadsPerBlock = THREADSPERBLOCK;

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

    __syncthreads(); // probably unnecessary, or probably should be __threadfence_block()

    // total score
    scores[index] = blockScore[0] + blockScore[1] + blockScore[2];


}
