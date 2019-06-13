
// System includes
#include <stdio.h>
#include <assert.h>
#include <chrono>
// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include <cuda_kerns.h>
#include <check_kerns.h>
#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif
#ifndef ARRSIZE
#define ARRSIZE 256*512*512 
#endif
int main(int argc, char **argv)
{
	int devID;
	cudaDeviceProp props;

	// This will pick the best possible CUDA capable device
	devID = findCudaDevice(argc, (const char **)argv);

	//Get GPU information
	checkCudaErrors(cudaGetDevice(&devID));
	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
	printf("Device %d: \"%s\" with Compute %d.%d capability\n",
		   devID, props.name, props.major, props.minor);

	printf("printf() is called. Output:\n\n");

	int* start = new int[4];
	int* stop = new int[4];
	int* mask = new int[4*64];
	start[0] = 15;
	start[1] = 16;
	start[2] = 31;
	start[3] = 30;
	stop[0] = 16; 
	stop[1] = 15; 
	stop[2] = 30; 
	stop[3] = 31; 
	for (int i=0; i<15;i++)
{		mask[i] = 1;
		mask[64+i] = 0;}
	mask[15] = 10;
	mask[64+15] = 0;
	mask[64+16] = 10;
	for (int i=17; i<64;i++)
	{	mask[i] = 0;
		mask[i+64] = 1;}

	for (int i=0; i<30;i++){
		mask[i+128] = 0;
		mask[i+192] = 1;
}
	mask[128+31] = 10;
	mask[192+30] = 10;
	mask[192+31] = 0;
	for (int i=32; i<64;i++){
		mask[128+i] = 1;
		mask[192+i] = 0;
}
	
	float *array = new float[ARRSIZE];
	float *arraycheck = new float[ARRSIZE];
	float *array2 = new float[ARRSIZE];
	float *array2check = new float[ARRSIZE];


//x
	for (int i =0; i<=15; i++)
	{
		array[i] = i+15.1;//i%64;
		arraycheck[i] =i+15.1;// i%64;
	}
	for (int i =16; i<=30; i++)
	{
		array[i] = 31.0;//i%64;
		arraycheck[i] =31.0;// i%64;
	}
	for (int i =31; i<64; i++)
	{
		array[i] = i+2.0;//i%64;
		arraycheck[i] =i+2.0;// i%64;
	}
	array[35] = 15;
	arraycheck[35] = 15;
//y

	for (int i =0; i<=15; i++)
	{
		array[64+i] = 15.0;//i%64;
		arraycheck[64+i] =15.0;// i%64;
	}
	for (int i =16; i<=30; i++)
	{
		array[64+i] = i+1.0;//i%64;
		arraycheck[64+i] =i+1.0;// i%64;
	}
	for (int i =31; i<64; i++)
	{
		array[64+i] = 30+2.0;//i%64;
		arraycheck[64+i] =30+2.0;// i%64;
	}
	array[64+35] = 15;
	arraycheck[64+35] = 15;

//z
	for (int i =0; i<64; i++)
	{
		array[i+128] = 15*1.0;//i%64;
		arraycheck[i+128] =15*1.0;// i%64;
	}
















	for (int i =0; i<ARRSIZE; i++)
	{
		array2[i] =0;//64-i;
		array2check[i] =0;//64-i;
	}

        float *volumeData = new (float [1000000]);
        float *ptr = volumeData;

//        for (size_t i=0; i<1000000; i++)
  //      {
    //        *ptr++ = i;
      //  }
		

//"pocket" init: 

	for (int x = 0; x<100;x++)
	{	
		int x_val = x<50 ? (x == 0 ? -1000 : x) : (x == 99 ? -1000 : 100 - x);
		for (int y = 0; y<100;y++)
		{	
			int y_val = y<50 ? (y == 0 ? -1000 : y) : (y == 99 ? -1000 : 100 - y);
			for (int z = 0; z<100;z++)
			{	
				int z_val = z<50 ? (z == 0 ? -1000 : z) : (z == 99 ? -1000 : 100 - z);
				volumeData[x+100*y+10000*z] = x_val+y_val+z_val;
			}
		}
	}















/*call function: angles to evaluate to 1°(on 256 total degrees).
 *
 * the function will have to evaluate data structures of 64 "atoms", that are the array (input data) and array2 (output data). 1 is the angle, volumeData is the value of the space for scoring, the start/stop arrays contains the position of the "atoms" used to initialize the rotation, and mask contains a bitmask that shows if an "atom" has to be rotated (1) or not (0). It also contains a 10 value that is the "start" atom.
 * functional details can be found inside the ps_check function, in ../include/check_kerns.h
 */
	ps_kern(array,array2,1,volumeData,start,stop,mask );





//check func
	ps_check<64,4>(arraycheck,array2check,1,volumeData,start,stop,mask );


	for (int i =0; i<64; i++)
		printf("%d:\t %.4f %.4f \t| %d: \t%.4f %.4f \t| %d: \t%.4f %.4f \n",i,array2[i],array2check[i],i+64,array2[1*64+i],array2check[1*64+i],i+128,array2[2*64+i],array2check[2*64+i]);
		//std::cout<<array2[i]<<"\t"<<array2check[i] <<" |\t"<<array2[1*64+i]<<"\t"<<array2check[1*64+i] <<" |\t"<<array2[2*64+i]<<"\t"<<array2check[2*64+i]<<std::endl;

		std::cout<<std::endl;

	delete[] array;
	delete[] arraycheck;
	delete[] array2;
	delete[] array2check;
	return EXIT_SUCCESS;
}