#ifndef LIMIT_DISTANCE2
#define LIMIT_DISTANCE2 = 2.0f;
#endif

//i pass is_bumping for thread from each block to access it
__device__ void fragment_is_bumping(const float* in, const int* mask, bool &is_bumping){

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
	        
	        //there could be multiple accesses to is_bumping! Should we avoid it? (WAW to same value, there should be no problem)
			if (distance2 < LIMIT_DISTANCE2) is_bumping = true;
	}
}
