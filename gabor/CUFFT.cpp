#include <stdlib.h>
#include "CUFFT.h"
#include <cufft.h>
#include <cuda_runtime_api.h>
#include <string.h>

CUFFT* CUFFT::_singleton = NULL;

CUFFT::CUFFT()
{
	_intialized = false;
}

CUFFT::~CUFFT()
{
	if (_intialized)
	{
		cudaThreadExit();
	}
}

CUFFT* CUFFT::_instance()
{
	if (_singleton == NULL)
		_singleton = new CUFFT();
	return _singleton;
}

void CUFFT::init()
{
	_instance()->_init();
}

void CUFFT::fft(float* data, int cols, int rows, bool forward)
{
	_instance()->_fft(data, cols, rows, forward);
}

void CUFFT::_init()
{
	int device_count = 0;
	cudaGetDeviceCount( &device_count );

	cudaDeviceProp device_properties;
	int max_gflops_device = 0;
	int max_gflops = 0;

	int current_device = 0;
	cudaGetDeviceProperties( &device_properties, current_device );
	max_gflops = device_properties.multiProcessorCount * device_properties.clockRate;
	++current_device;

	while( current_device < device_count )
	{
		cudaGetDeviceProperties( &device_properties, current_device );
		int gflops = device_properties.multiProcessorCount * device_properties.clockRate;
		if( gflops > max_gflops )
		{
			max_gflops        = gflops;
			max_gflops_device = current_device;
		}
			++current_device;
	}

	cudaSetDevice(max_gflops_device);
	_intialized = true;
}

void CUFFT::_fft(float* data, int cols, int rows, bool forward)
{
	cufftHandle plan;
	cufftComplex* gpuData;
	cudaMalloc((void**)&gpuData, sizeof(cufftComplex) * cols * rows);
	cufftPlan2d(&plan, cols, rows, CUFFT_C2C);
	cudaMemcpy(gpuData, data, sizeof(cufftComplex) * cols * rows, cudaMemcpyHostToDevice);
	cufftExecC2C(plan, gpuData, gpuData, forward? CUFFT_FORWARD : CUFFT_INVERSE);
	cufftDestroy(plan);
	cudaMemcpy(data, gpuData, sizeof(cufftComplex) * cols * rows, cudaMemcpyDeviceToHost);
	cudaFree(gpuData);
}
