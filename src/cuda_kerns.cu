
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
__global__ void measure_shotgun(float* d_in, float* d_score_pos, int* d_shotgun);
__global__ void fragment_is_bumping(float* d_in, int* d_mask, bool* d_bumping, int i);
__global__ void eval_angles(float* in, float* score_pos, int& best_angle, int& best_score, free_rotation::value_type& best_rotation_matrix, int start_index, int stop_index, int* mask);
__global__ void update(float* in, float* out);


void ps_kern(float* in, float* out, int precision, float* score_pos, int* start, int* stop, int* mask )
{
	float *d_in, *d_score_pos, *d_rotation_matrix;
	int *d_start, *d_stop, *d_mask;

	cudaError_t status, status_cp, status_tx;
	
	//Dimensions of block and grid for the 'fragment_is_bumping' functions
	dim3 bumping_block(INSIZE,1,1); //256,1,1
	dim3 bumping_grid(INSIZE,MAX_ANGLE*1/precision,1); //256,256,1

	//GPU MEMORY INITIALIZATION
	textue<float> texScore_pos;
	texture<int> texMask;
	
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


	status = cudaMalloc((void**) &d_rotation_matrix, sizeof(float)*12);
  if(DEBUG && status!=cudaSuccess)
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status_cp = cudaMemcpy(d_rotation_matrix, rotation_matrix, sizeof(int)*12, cudaMemcpyHostToDevice);
	if(DEBUG && status_cp!=cudaSuccess)
		cout << cudaGetErrorString(status_cp) << " in " << __FILE__ << " at line " << __LINE__ << endl;	

  // --------------
	status_tx = cudaBindtexture(NULL, texScore_pos, d_score_pos, sizeof(float)*VOLUMESIZE);
	if(DEBUG && status_tx!=cudaSuccess)
	  cout << cudaGetErrorString(status_tx) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status_tx = cudaBindtexture(NULL, texMask, d_mask, sizeof(float)*VOLUMESIZE);
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

		const auto epsilon = std::numeric_limits<float>::epsilon();

		// get the index of starting atom
		const auto start_atom_index = start[i];
		const auto stop_atom_index = stop[i];

		int best_angle = 0;
		float best_score = 0;
		free_rotation::value_type best_rotation_matrix; // right syntax?

		eval_angles<<<256*ceil(1/precision),1>>>(in, score_pos, best_angle, best_score, best_rotation_matrix, start_atom_index, stop_atom_index, mask);

		std::cout<<"best angle is: "<<best_angle<<std::endl;
		std::cout<<" score is: "<<best_score<<" for fragm: "<<i<<"with angle out"<<std::endl;

	}

	//stop CUDA timing here
		
	cudaError_t status_wb;
	//we (should) copy the best 'in' array back in central memory as the 'out' array 
	//(this memcpy copies from the gpu the FIRST 'in' array (index 0-255) that isn't necessarily the best, unless "compute_best" ensures this)
	status_wb = cudaMemcpy(out, d_in, sizeof(float)*ARRSIZE, cudaMemcpyDeviceToHost);
	if(DEBUG && status_wb!=cudaSuccess)
		cout << cudaGetErrorString(status_wb) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	cudaUnbindTexture(texScore_pos);
	cudaUnbindTexture(texMask);
	cudaFree(d_score_pos);
	cudaFree(d_start);
	cudaFree(d_stop);
	cudaFree(d_mask);
	cudaFree(d_rotation_matrix);
	cudaFree(d_in);
