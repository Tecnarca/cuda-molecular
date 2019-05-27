#ifndef DEBUG
#define INSIZE 256 //real (used by the main.cpp) dimension of 'in'
#define N_ATOMS 64 //atoms inside 'in'
#define N_FRAGS 4 //number of fragments
#define DEBUG 1 //should we execute the debug prints and checks?
#define MASKSIZE 256 //dimension of the 'mask'
#define VOLUMESIZE 1000000 //dimension of 'score_pos'
#define THREADSPERBLOCK 512 //how many threads per block? (max 1024) <- probabilmente c'è da vedere quanto conviene lasciarlo così
#define BLOCKS 3 //how many blocks per grid? <- probabilmente c'è da vedere quanto conviene lasciarlo così
#define MAX_ANGLE 256 //up to which angle we need to run the algorithm?
#define LIMIT_DISTANCE2 = 2.0f; //used in fragment_is_bumping, it is the minimum distance between to atoms
#endif

//Speed improvements:
//0) Implement CONSTANT and TEXTURE memory effectively
//1) (having n_atoms as global) remove the indexing by "+blockDim.x" in the kernels to use "+N_ATOMS", that is a predefined variable (so it's faster to access?)
//2) If we want to have a greater parallelism, we could put the malloc and the memcpy on different streams (not requested for this project by the professor)
//3) Check if it's convenient to remove the 'i' parameter from some kernels and use the '&mask[i*N_ATOMS]' instead
//4) [put other ideas to test later here]


//CUDA KERNELS DECLARATION
__global__ void rotateN(float* d_in, int* d_mask, int precision, int i);
__global__ void measure_shotgun(float* d_in, float* d_score_pos, int* d_shotgun);
__global__ void fragment_is_bumping(float* d_in, int* d_mask, bool* d_bumping, int i);
__global__ void compute_best(float* d_in, int* d_shotgun, bool* d_bumping);


void ps_kern(float* in, float* out, int precision, float* score_pos, int* start, int* stop, int* mask )
{
	float *d_in, *d_score_pos, *d_rotation_matrix;

	int *d_start, *d_stop, *d_mask, *d_shotgun;

	bool *d_bumping;

	cudaError_t status, status_cp;

	//Dimensions of block and grid for the 'fragment_is_bumping' functions
	dim3 bumping_block(INSIZE,1,1); //256,1,1
	dim3 bumping_grid(INSIZE,MAX_ANGLE*1/precision,1); //256,256,1

	//GPU MEMORY INITIALIZATION

	/*'in' on the GPU is long INSIZE*256*ceil(precision) so that we can save all the ROTATED 'in' arrays at the same time.
	The first INSIZE cells are the original 'in' array (and we initialize only those with memcpy)*/
	status = cudaMalloc((void**) &d_in, sizeof(float)*INSIZE*MAX_ANGLE*ceil(1/precision));
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


	status = cudaMalloc((void**) &d_start, sizeof(int)*N_ATOMS);
	if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status_cp = cudaMemcpy(d_start, start, sizeof(int)*N_ATOMS, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		cout << cudaGetErrorString(status_cp) << " in " << __FILE__ << " at line " << __LINE__ << endl;


	status = cudaMalloc((void**) &d_stop, sizeof(int)*N_ATOMS);
    if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status_cp = cudaMemcpy(d_stop, stop, sizeof(int)*N_ATOMS, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		cout << cudaGetErrorString(status_cp) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status = cudaMalloc((void**) &d_mask, sizeof(int)*MASKSIZE);
    if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status_cp = cudaMemcpy(d_mask, mask, sizeof(int)*MASKSIZE, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		cout << cudaGetErrorString(status_cp) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	//'bumping' is an array that for each 'in' array tells us if the fragment is bumping
	status = cudaMalloc((void**)d_bumping, sizeof(bool)*MAX_ANGLE*ceil(1/precision));
	if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	//these lines initializes the d_bumping array to false
	status_cp = cudaMemset((void**)d_bumping, 0, sizeof(bool)*MAX_ANGLE*ceil(1/precision));
	if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;


	//'shotgun' is an array that for each 'in' array tells us what the score of that array is
	status = cudaMalloc((void**)d_shotgun, sizeof(int)*MAX_ANGLE*ceil(1/precision));
	if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status = cudaMalloc((void**) &d_rotation_matrix, sizeof(float)*MASKSIZE);
    if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	//CUDA stream creation
	//https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf
	//The webminar above explains how to use effectively the cuda streams
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	// start CUDA timing here

	for (int i=0;i<n_frags;++i){ //Rotameter optimization. Numbers after the cells is the stream channel.

		//EACH FUNCTION in this 'for' must be called as a KERNEL (decide sizes)
		//also: stream 0 means that this kernel starts execution only when there are no other streams running (classic kernel calls are put in stream 0)

		//PUT GLOBAL SYNC HERE: Qual'è la funzione che sincronizza tutti gli stream?	

		/*this function takes in input the first 'in' array and computes every matrix (for each angle) 
		and writes in the other 'in' arrays all the rotated arrays.
		From the sequestial program perspective, this function:
		>computes start_atom_index and stop_atom_index for the 3 axis
		>executes compute_matrix()
		>executes rotate()
		The 'precision' parameter is needed when it is not 1 (it is for us, we keep it in order to keep this code general (?)):
		Is needed in order to create an associativity between 'in' vectors and the rotations (we will see this better inside the rotateN function when we'll write it [nota per marco: significa che non sono sicuro che questa cosa sia giusta, ma ad occhio si])
		The 'i' parameter is needed, since the compute_matrix() function needs to know which fragment must be used to compute the rotation mask 
		(^ can this be done by passing &d_mask[i*n_atoms]? If so, 'i' is useless)*/
		rotateN<<<?,?,0,stream1>>>(d_in, d_mask, precision, i); //must be put on stream 1

		//this initialization is needed by fragment_is_bumping and is completely in parallel w.r.t rotateN() (rotateN does not need d_bumping, so we reset it here)
		status_cp = cudaMemsetAsync((void**)d_bumping, 0, sizeof(bool)*MAX_ANGLE*ceil(1/precision), stream2);
		if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;


		//PUT GLOBAL SYNC HERE: Qual'è la funzione che sincronizza tutti gli stream?	

		/*For each 'in' vector (every thread must read a different 'in' vector), we must compute the shotgun and place it in the shotgun vector, everyone at the corresponding index of the 'in' array
		e.g. the first thread (or the first block, depending on the setting) reads the first 'in' array (positions 0-255) and places the result in shotgun[0],
		the second thread (or the first block, depending on the setting) reads the second 'in' array (positions 256-511) and places the result in shotgun[1] etc.
		per marco: se hai problemi a realizzare questa cosa, è possibile che devi usare una griglia,
		oppure mettere i blocchi in 2d (probabilmente dovrò fare la stessa cosa su fragment_is_bumping() */
		measure_shotgun<<<?,?, 0, stream1>>>(d_in, d_score_pos, d_shotgun); //must be put on stream 1

		/*As for measure_shotgun, for each 'in' vector, we must compute if the fragment is bumping and place the result in the 'bumping' vector.
		This operation can be performed completely in parallel w.r.t the measure_shotgun kernels (same reads, no input modification, the outputs are on different variables)
		The 'i' parameter is needed, since the compute_matrix() function needs to know which fragment must be used to compute the rotation mask 
		(^ can this be done by passing &d_mask[i*n_atoms]? If so, 'i' is useless) */
		fragment_is_bumping<<<bumping_grid,bumping_block, 0, stream2>>>(d_in, d_mask, d_bumping, i); //must be put on stream 2

		//PUT GLOBAL SYNC HERE: Qual'è la funzione che sincronizza tutti gli stream?

		/*Starting from all the 'in' array and given their scores and if they are bumping:
		this function extracts which 'in' array is the best one and copies it back as the first array, 
		to start the next computation from that one.
		*/
		compute_best<<<?,?>>>(d_in, d_shotgun, d_bumping); //must be put on stream 0

	}

	//stop CUDA timing here
		
	cudaError_t status_wb;
	//we (should) copy the best 'in' array back in central memory as the 'out' array 
	//(this memcpy copies from the gpu the FIRST 'in' array (index 0-255) that isn't necessarily the best, unless "compute_best" ensures this)
	status_wb = cudaMemcpy(out, d_in, sizeof(float)*ARRSIZE, cudaMemcpyDeviceToHost);
	if(DEBUG && status_wb!=cudaSuccess)
		cout << cudaGetErrorString(status_wb) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	// none of these data structures is changed by any kernel, thus we can texturize
	cudaFree(d_score_pos);
	cudaFree(d_start);
	cudaFree(d_stop);
	cudaFree(d_mask);

    // not texturizable
	cudaFree(d_in);

}