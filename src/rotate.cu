// --- THIS PART TO BE INCLUDED IN cuda_kerns.h ---
// __constant__ float rotation_matrix[9]

// --- END ---



// --- THIS PART TO BE INCLUDED IN PS_KERN ---
// cudaError_t status;

// status = cudaMemcpyToSymbol(dev_rotation_matrix, rotation_matrix, 9*sizeof(float))

// if(status != cudaSuccess) {
//	 cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
// }
// --- END ---

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
		float m11 = rotation_matrix[0];
		float m12 = rotation_matrix[1];
		float m13 = rotation_matrix[2];
		float m14 = rotation_matrix[3];
		float m21 = rotation_matrix[4];
		float m22 = rotation_matrix[5];
		float m23 = rotation_matrix[6];
		float m24 = rotation_matrix[7];
		float m31 = rotation_matrix[8];
		float m32 = rotation_matrix[9];
		float m33 = rotation_matrix[10];
		float m34 = rotation_matrix[11];

    // update in (the one that is allocated in the GPU)
		in[x] = m11 * prev_x + m12 * prev_y + m13 * prev_z + m14;
		in[y] = m21 * prev_x + m22 * prev_y + m23 * prev_z + m24;
		in[z] = m31 * prev_x + m32 * prev_y + m33 * prev_z + m34;

}
