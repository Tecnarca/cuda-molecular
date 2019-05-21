#ifndef ARRSIZE
#define ARRSIZE 256*512*512 
#endif
#ifndef DEBUG
#define DEBUG 1
#define NATOM 4
#define MASKSIZE 256
#define VOLUMESIZE 1000000
#endif

extern "C" void ps_kern(float* atoms_in, float* atoms_out, int precision,float* score_pos, int* start, int* stop, int* mask )
{
	float *d_atoms_in, *d_score_pos;
	
	//do we need these in the gpu?
	int* d_start, int* d_stop, int* d_mask;
	
	//we do not need atoms_out and precision on the gpu

	cudaError_t status;

	status = cudaMalloc((void**) &d_atoms_in, siezof(float)*ARRSIZE);
	
	if(DEBUG && status!=cudaSuccess) 
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status = cudaMalloc((void**) &d_score_pos, siezof(float)*VOLUMESIZE);
	
	if(DEBUG && status!=cudaSuccess) 
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status = cudaMalloc((void**) &d_start, siezof(float)*NATOM);
	
	if(DEBUG && status!=cudaSuccess) 
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status = cudaMalloc((void**) &d_stop, siezof(float)*NATOM);
	
	if(DEBUG && status!=cudaSuccess) 
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	status = cudaMalloc((void**) &d_mask, siezof(float)*MASKSIZE);
	
	if(DEBUG && status!=cudaSuccess) 
		cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;

	


}
