#include <stdio.h>

static __global__ void generateFilter(float* filter, 
                                      float lambda, 
									  float theta, 
									  float psi, 
									  float sigma, 
									  float gamma,
									  int2 center,
									  int2 size)
{
	int xCoord = blockIdx.x * blockDim.x + threadIdx.x;
	int yCoord = blockIdx.y * blockDim.y + threadIdx.y;
	float x = xCoord - center.x;
	float y = yCoord - center.y;
	float xPrime = x * cos(theta) + y * sin(theta);
	float yPrime = -x * sin(theta) + y * cos(theta);
	float harmonic = 2.0 * M_PI * xPrime / lambda + psi;
	float exponential = -(xPrime * xPrime + gamma * gamma * yPrime * yPrime) /
	                    (2.0 * sigma * sigma);
	filter[yCoord * size.x + xCoord] = exp(exponential) * cos(harmonic);
}

static __global__ void gaussianKernel(float* filter,
                                      float theta,
                                      float sigma,
								      float gamma,
								      int2 center,
								      int2 size)
{
	int xCoord = blockIdx.x * blockDim.x + threadIdx.x;
	int yCoord = blockIdx.y * blockDim.y + threadIdx.y;
	float x = xCoord - center.x;
	float y = yCoord - center.y;
	float xPrime = x * cos(theta) + y * sin(theta);
	float yPrime = -x * sin(theta) + y * cos(theta);
	float exponent = -(xPrime * xPrime + gamma * gamma * yPrime * yPrime) /
	                 (2.0 * sigma * sigma);
	filter[yCoord * size.x + xCoord] = exp(exponent) / (2.0 * M_PI * sigma * sigma);
}

extern "C"
void gaussian(float* filter, float theta, float sigma, float gamma, int2 center, int2 size)
{
	int filterSize = size.x;
	dim3 blockDim(64, 64);
	dim3 gridDim(filterSize / 64, filterSize / 64);
	gaussianKernel<<<blockDim, gridDim>>>(filter, theta, sigma, gamma, center, size);
}

static __global__ void harmonicKernel(float* filter,
                                      float theta,
									  float lambda,
									  float psi,
									  int2 center,
									  int2 size)
{
	int xCoord = blockIdx.x * blockDim.x + threadIdx.x;
	int yCoord = blockIdx.y * blockDim.y + threadIdx.y;
	float x = xCoord - center.x;
	float y = yCoord - center.y;
	float xPrime = x * cos(theta) + y * sin(theta);
	float angle = 2.0 * M_PI * xPrime / lambda + psi;
	filter[(yCoord * size.x + xCoord) * 2] = cos(angle);
	filter[(yCoord * size.x + xCoord) * 2 + 1] = sin(angle);
}

extern "C"
void harmonic(float* filter, float theta, float lambda, float psi, int2 center, int2 size)
{
	int filterSize = size.x;
	dim3 blockDim(64, 64);
	dim3 gridDim(filterSize / 64, filterSize / 64);
	harmonicKernel<<<blockDim, gridDim>>>(filter, theta, lambda, psi, center, size);
}

static __global__ void multiplyRealComplexKernel(float* real, float* complex, float* result)
{
	int realIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int complexIndex = realIndex * 2;
	result[complexIndex] = real[realIndex] * complex[complexIndex];
	result[complexIndex + 1] = real[realIndex] * complex[complexIndex + 1];
	//result[complexIndex] = real[realIndex];
	//result[complexIndex + 1] = 0;

}

extern "C"
void multiplyRealComplex(float* real, float* complex, float* result, int numElements)
{
	multiplyRealComplexKernel<<<numElements / 32, 32>>>(real, complex, result);
}

static __global__ void multiplyComplexComplexKernel(float* a, float* b, float* result)
{
	int realIndex = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int imagIndex = realIndex + 1;
	float areal = a[realIndex];
	float acomplex = a[imagIndex];
	float breal = b[realIndex];
	float bcomplex = b[imagIndex];
	result[realIndex] = (areal * breal) - (acomplex * bcomplex);
	result[imagIndex] = (areal * bcomplex) + (breal * acomplex);
	//result[realIndex] = breal;
	//result[imagIndex] = bcomplex;
}

extern "C"
void multiplyComplexComplex(float* a, float* b, float* result, int numElements)
{
	multiplyComplexComplexKernel<<<numElements / 64, 64>>>(a, b, result);
}

static __global__ void centerKernel(float* data, int2 size)
{
	uint xCoord = blockIdx.x * blockDim.x + threadIdx.x;
	uint yCoord = blockIdx.y * blockDim.y + threadIdx.y;
	if ((xCoord + yCoord) & 1u)
	//if ((xCoord + yCoord) % 2 == 1)
	{
		data[(yCoord * size.x + xCoord) * 2] *= -1;
		data[(yCoord * size.x + xCoord) * 2 + 1] *= -1;
	}
}

extern "C"
void center(float* data, int2 size)
{
	dim3 blockDim(32, 32);
	dim3 gridDim(size.x / 32, size.y / 32);
	centerKernel<<<blockDim, gridDim>>>(data, size);
}

static __global__ void complexToMagnitudeKernel(float* complex, float* magnitude)
{
	int realIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int complexIndex = realIndex * 2;
	float real = complex[complexIndex];
	float imag = complex[complexIndex + 1];
	magnitude[realIndex] = sqrt(real * real + imag * imag) * 0.5;
}

extern "C"
void complexToMagnitude(float* complex, float* magnitude, int numElements)
{
	complexToMagnitudeKernel<<<32, numElements / 32>>>(complex, magnitude);
}

static __global__ void complexToRealKernel(float* complex, float* real)
{
	int realIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int complexIndex = realIndex * 2;
	real[realIndex] = complex[complexIndex];
}

extern "C"
void complextoReal(float* complex, float* real, int numElements)
{
	complexToRealKernel<<<32, numElements / 32>>>(complex, real);
}

static __global__ void realToComplexKernel(float* real, float* complex)
{
	int realIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int complexIndex = realIndex * 2;
	complex[complexIndex] = real[realIndex];
	complex[complexIndex + 1] = 0.0;
}

extern "C"
void realToComplex(float* real, float* complex, int numElements)
{
	realToComplexKernel<<<32, numElements / 32>>>(real, complex);
}

//takes the absolute difference and normalizes assuming that the range is [0,512]
static __global__ void differenceImagesKernel(float* a, float* b, float* result, float scale)
{
	int index = (blockIdx.x * blockDim.x + threadIdx.x)*2;	
	//result[index] = abs( a[index] - b[index] ) * (1.0 / 512.0) * scale;
	//result[index] = abs( a[index] - b[index] ) * (1.0 / 255.0) * scale;
	result[index] = abs( a[index] - b[index] ) * (1.0f / 255.0f);
	//result[index] = abs( a[index]);
	result[index + 1] = 0.0f;
}

extern "C"
void differenceImages(float* a, float* b, float* result, int numElements)
{
	differenceImagesKernel<<<numElements / 64, 64>>>(a, b, result, (1.0f / (float)numElements));
}

static __global__ void computeAveragesKernel(float* a, float* result, int2 size, int2 pSize, int2 offset, float scale)
{
	extern __shared__ float tmpArr[];

	float total = 0.0f;
	int tmpArrIndex = threadIdx.x;
	int startIndex = 2*( pSize.x * (offset.y + size.y / gridDim.y * blockIdx.y + threadIdx.x) + offset.x + (size.x / gridDim.x * blockIdx.x) );
	int endIndex = startIndex + 2*(size.x / gridDim.x);

	for(int i=startIndex; i<endIndex; i+=2)
		//total += a[i];
		total += abs(a[i]);
	// Save partial average
	tmpArr[tmpArrIndex] = total / (float)(size.x / gridDim.x);

	__syncthreads();

	if( threadIdx.x == 0 )
	{
		int tmpArrSize = blockDim.x;
		total = 0.0f;
		for(int i=0; i<tmpArrSize; i++)
		{
			total += tmpArr[i];
		}

		// Save average of partial averages
		//the index is for the smaller result grid
		float r = total * scale / (float)tmpArrSize;
		result[gridDim.x*blockIdx.y + blockIdx.x] = r < 0.0f? 0.0f : r > 1.0f? 1.0f: r; 
		//result[gridDim.x*blockIdx.y + blockIdx.x] = 0.5f + 0.5f * scale * total / (float)tmpArrSize;
	}
}

static __global__ void computeMinMaxKernel(float* a, float* minMax, int2 divFactor)
{
	extern __shared__ float tmpMinMaxArr[];

	int index = threadIdx.x * divFactor.x;
	int endIndex = index + divFactor.x;
	float min = a[index];
	float max = min;
	for(int i = index; i<endIndex; ++i)
	{
		if( a[i] < min )
			min = a[i];
		if( a[i] > max )
			max = a[i];
	}
	tmpMinMaxArr[threadIdx.x*2] = min;
	tmpMinMaxArr[threadIdx.x*2+1] = max;

	__syncthreads();

	if( threadIdx.x == 0 )
	{
		min = tmpMinMaxArr[0];
		max = min;
		for(int i = 0; i<blockDim.x*2; ++i)
		{
			if( tmpMinMaxArr[i] < min )
				min = tmpMinMaxArr[i];
			if( tmpMinMaxArr[i] > max )
				max = tmpMinMaxArr[i];
		}
		minMax[0] = min / 256.0;
		minMax[1] = max / 256.0;
	}
}

static __global__ void computeMinMaxKernel2(float* a, float* minMax, int2 offset, int2 imageSize, int2 paddedSize)
{
	extern __shared__ float tmpMinMaxArr[];

	int index = ((threadIdx.x + offset.y) * paddedSize.x + offset.x) * 2;
	int endIndex = index + imageSize.x * 2;
	float min = a[index];
	float max = min;
	for(int i = index; i<endIndex; i+=2)
	{
		if( a[i] < min )
			min = a[i];
		if( a[i] > max )
			max = a[i];
	}
	tmpMinMaxArr[threadIdx.x*2] = min;
	tmpMinMaxArr[threadIdx.x*2+1] = max;

	__syncthreads();

	if( threadIdx.x == 0 )
	{
		min = tmpMinMaxArr[0];
		max = min;
		for(int i = 0; i<blockDim.x*2; ++i)
		{
			if( tmpMinMaxArr[i] < min )
				min = tmpMinMaxArr[i];
			if( tmpMinMaxArr[i] > max )
				max = tmpMinMaxArr[i];
		}
		minMax[0] = min / 256.0 / 256.0;
		minMax[1] = max / 256.0 / 256.0;
	}
}

static __global__ void normalizeAveragesKernel(float* result, float* minMax)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;	
	result[index] = (result[index] - minMax[0]) / (minMax[1] - minMax[0]);
}

extern "C"
void computeProbabilities(float* a, float* result, int2 divFactor, int2 size, int2 pSize, int2 offset)
{
	printf("%d %d\n", pSize.x, pSize.y);
	int threadsPerBlock = size.y / divFactor.y;
	dim3 gridDim(divFactor.x, divFactor.y);
	// Compute the averages for each of the regions
	int sharedMemSize = sizeof(float) * size.y / divFactor.y;
	static float globalMin = 10e100;
	static float globalMax = -10e100;
	static float uselessAverage = 0.0f;
	static int count = 0;

#if 0 // print values to be averaged
	float* vals = new float[pSize.x * pSize.y * 2];
	cudaMemcpy(vals, a, sizeof(float) * pSize.x * pSize.y * 2, cudaMemcpyDeviceToHost);
	for(int i=0; i<pSize.x * pSize.y; i+=2)
		if( vals[i] > 20000.0f )
			printf("val[%d]: %f\n", i, vals[i]);	
#endif

#if 0 // print averages
	{
                // Calculate averages on host (for comparison)
                float* h = new float[divFactor.x * divFactor.y];
                float* vals = new float[pSize.x * pSize.y * 2];
                int blockStride = size.x / divFactor.x;
                int blockHeight = size.y / divFactor.y;
//              int valsPerRegion = size.x / divFactor.x * size.y / divFactor.y;
                cudaMemcpy(vals, a, sizeof(float) * pSize.x * pSize.y * 2, cudaMemcpyDeviceToHost);
                for(int i=0; i<divFactor.x; ++i)
                        for(int j=0; j<divFactor.y; ++j)
                        {
                                int startIndex = pSize.x * offset.y + offset.x; 
                                startIndex += pSize.x * blockHeight * j;
                                startIndex += blockStride * i;
                                startIndex *= 2;
				
                                int hIndex = divFactor.x * j + i;
                                h[hIndex] = 0.0f;
                                float pAvg = 0.0f;
                                for(int k=0; k<blockHeight; ++k)
                                {
					// Calculate and save partial average
                                        pAvg = 0.0f;
                                        for(int z=0; z<blockStride*2; z+=2)
                                        {
                                                pAvg += vals[startIndex + z];
                                        }
                                        startIndex += 2 * pSize.x;
                                        h[hIndex] += pAvg / (float) blockStride;
                                }       
				// Calculate average of partial averages
                                h[hIndex] /= (float)blockHeight;
                        }

		float* tmp = new float[divFactor.x * divFactor.y];	
		cudaMemcpy(tmp, result, sizeof(float)*divFactor.x*divFactor.y, cudaMemcpyDeviceToHost);
		// print out to compare: averages w/ CUDA and averages on Host (CPU)
		for(int i=0; i<divFactor.x*divFactor.y; ++i)
			printf("Avg[%d]: %f : %f %s\n", i, tmp[i], h[i], (tmp[i]==h[i]?"":"**MISMATCH**") );	
		printf("\n");
		delete tmp;
		delete h;
		delete vals;
	}
#endif

	// Compute the min/max of the averages
	float* minMax;
	cudaMalloc((void**)&minMax, sizeof(float)*2);
	//sharedMemSize = sizeof(float)*divFactor.y*2;
	//computeMinMaxKernel<<<1, divFactor.y, sharedMemSize>>>(result, minMax, divFactor);
	sharedMemSize = sizeof(float)*size.y*2;
	//computeMinMaxKernel<<<1, size.y, sharedMemSize>>>(a, minMax, size);
	computeMinMaxKernel2<<<1, size.y, sharedMemSize>>>(a, minMax, offset, size, pSize);

#if 1 // print min/max
	float shittyScalar;
	{
		//float* avgs = new float[divFactor.x * divFactor.y];	
		//cudaMemcpy(avgs, result, sizeof(float)*divFactor.x*divFactor.y, cudaMemcpyDeviceToHost);
		//int vals = divFactor.x * divFactor.y;
		//float min = avgs[0];
		//float max = avgs[0];
		//for(int i=0; i<vals; ++i)
		//{
		//	if( avgs[i] < min ) min = avgs[i];	
		//	if( avgs[i] > max ) max = avgs[i];	
		//}
		float tmp[2];
		cudaMemcpy( tmp, minMax, sizeof(float)*2, cudaMemcpyDeviceToHost);
		//printf("%f %f\n", tmp[0], tmp[1]);
		globalMin = tmp[0] < globalMin? tmp[0]: globalMin;
		globalMax = tmp[1] > globalMax? tmp[1]: globalMax;
		shittyScalar = tmp[1] / uselessAverage;
		uselessAverage = (uselessAverage * count + tmp[1]) / (count + 1);
		++count;
		printf("%f %f\n", globalMin, globalMax);
		tmp[0] = globalMin;
		tmp[1] = globalMax;
		cudaMemcpy(minMax, tmp, sizeof(float) * 2, cudaMemcpyHostToDevice);
		//printf("Min/Max: %f / %f : %f %f %s\n\n", tmp[0], tmp[1], min, max,
		//	(tmp[0]==min&&tmp[1]==max?"":"**MISMATCH**") );
		//delete avgs;
	}
#endif

	// Normalize each average based on min/max of the averages
	int nBlocks, blockSize;
	if( divFactor.x * divFactor.y < 4 )
	{
		nBlocks = 1;
		blockSize = divFactor.x * divFactor.y;
	}
	else // assuming that divFactor.x * divFactor.y is divisible by 4
	{
		nBlocks = divFactor.x * divFactor.y / 4;
		blockSize = divFactor.x * divFactor.y / nBlocks;
	}
	computeAveragesKernel<<<gridDim, threadsPerBlock, sharedMemSize>>>(a, result, size, pSize, offset, 1.0f / 256.0f / 256.0f);
	//normalizeAveragesKernel<<<nBlocks, blockSize>>>(result, minMax);
	cudaFree(minMax);
}

