/*
* File name: WorkerThread.cpp
*/

#include <QDebug>
#include <cuda_runtime.h>
#include "WorkerThread.h"

extern "C"
void gaussian(float* filter, float theta, float sigma, float gamma, int2 center, int2 size);
extern "C"
void harmonic(float* filter, float theta, float lambda, float psi, int2 center, int2 size);
extern "C"
void multiplyRealComplex(float* real, float* complex, float* result, int numElements);
extern "C"
void multiplyComplexComplex(float* a, float* b, float* result, int numElements);
extern "C"
void center(float* data, int2 size);
extern "C"
void complexToMagnitude(float* complex, float* magnitude, int numElements);
extern "C"
void complextoReal(float* complex, float* real, int numElements);
extern "C"
void realToComplex(float* real, float* complex, int numElements);
extern "C"
void differenceImages(float* a, float* b, float* result, int numElements);

// Self-Throttling Timer
#define THROTTLE_TIMER_ID 13
volatile sig_atomic_t g_throttleTimerFlag;
struct sigevent g_signalSpec;
timer_t g_timerID;
struct itimerspec g_throttleTimerSetting;

void throttleTimerHandler(int sig)
{
	g_throttleTimerFlag = 1;
}

// Default constructor.
WorkerThread::WorkerThread(QString video_source, QObject *parent) : QThread(parent)
{
	_original_image_width = 320;
	_original_image_height = 240;
	_is_image_processed = true;
	_camera = new V4LCamera(video_source.toStdString().c_str(), _original_image_width, _original_image_height);

	_padded_pixels = 0;
	_target_x = _original_image_width / 2;
	_target_y = _original_image_height / 2;
	_sigma = 9.0;
	_lambda = 25.0;
	_padded_image = NULL;
	_filter_image = NULL;
	_gpu_image_0 = NULL;
	_gpu_image_1 = NULL;
	_curr_gpu_image = 0;
	_gabor_data = NULL;

	_do_difference_images = true;

	_padded_size = _filter_size + _target_size;
	int log = 0;
	while (_padded_size != 1)
	{
		_padded_size >>= 1;
		++log;
	}
	_padded_size = 1 << log;
	_padded_pixels = _padded_size * _padded_size;
	_padded_image = new float[_padded_pixels * 2];
	_filter_image = new float[_padded_pixels * 2];
	_result_data = new float[_filter_pixels * 2];
	_host_gabor_data = new float[_filter_pixels * 2];

	_cleanup_mutex.lock();
	_should_terminate = false;
	_cleanup_mutex.unlock();

	//*** Throttling
	g_throttleTimerFlag = 1;
	int err;
	signal( SIGUSR1, throttleTimerHandler );
	
	// setup the timer specification
	g_signalSpec.sigev_notify = SIGEV_SIGNAL;
	g_signalSpec.sigev_signo = SIGUSR1;
	g_signalSpec.sigev_value.sival_int = THROTTLE_TIMER_ID;
	
	// create the timer
	err = timer_create(CLOCK_REALTIME, &g_signalSpec, &g_timerID);
	if( err < 0 )
	{
		qDebug() << "Error creating self-throttling timer";
		exit(0);
	}
	
	// Set the timer (and start the timer)
	g_throttleTimerSetting.it_value.tv_sec = 0;
	g_throttleTimerSetting.it_value.tv_nsec = 1e8;
	g_throttleTimerSetting.it_interval.tv_sec = 0;
	g_throttleTimerSetting.it_interval.tv_nsec = 1e8;
	err = timer_settime( g_timerID, 0, &g_throttleTimerSetting, NULL );
	if( err < 0 )
	{
		qDebug() << "Error setting throttling timer";
		exit(0);
	}
}

WorkerThread::~WorkerThread()
{
	delete _padded_image;
	delete _filter_image;
	delete _result_data;
	delete _host_gabor_data;
}

void WorkerThread::cleanup()
{
	_cleanup_mutex.lock();
	_should_terminate = true;
	_cleanup_mutex.unlock();
}

// Main execution loop.
// Capture from camera, filter image data, save filered data.
// Note*: wait conditions are used to ensure that the previous image
// has been rendered while allowing for continued operation.
void WorkerThread::run()
{
	createInitialFilter();
	while( 1 )
	{
		// Throttling
		while( !g_throttleTimerFlag ){}
		g_throttleTimerFlag = 0;

		// Capture image data
		_camera->capture();
		float *data = _camera->frameData();

		// Need a new filter?
		_new_filter_mutex.lock();	
		if( _should_create_new_filter )
		{
			createNewFilter();
		}
		_new_filter_mutex.unlock();	

		// Gabor filter
		gaborFilter( data );	

		// Make sure the previous image has been rendered before overwriting it
		_image_mutex.lock();
		if( !_is_image_processed )
		{
			_image_processed.wait(&_image_mutex);	
		}

		// Copy image data so the thread continues capturing and filtering
		memcpy( _full_image, data, sizeof(float) * _original_image_width * _original_image_height * 2 );
		_is_image_processed = false;
		emit filterComplete();
		_image_mutex.unlock();

		if( _should_terminate )
		{
			break;
		}
	}

	// Free GPU resources
	cudaFree(_gabor_data);
	cudaFree(_gpu_image_0);
	cudaFree(_gpu_image_1);
	cufftDestroy(_fft_plan);
}

void WorkerThread::createInitialFilter()
{
	float* gaussian_data;
	cudaMalloc((void**)&gaussian_data, sizeof(float) * _filter_pixels);
	int2 gaussian_size;
	gaussian_size.x = _filter_size;
	gaussian_size.y = _filter_size;
	int2 gaussian_center;
	gaussian_center.x = _filter_size / 2;
	gaussian_center.y = _filter_size / 2;
	gaussian(gaussian_data, 0.0, _sigma, 1.0, gaussian_center, gaussian_size);
	
	float* harmonic_data;
	cudaMalloc((void**)&harmonic_data, sizeof(float) * _filter_pixels * 2);
	int2 harmonic_size;
	harmonic_size.x = _filter_size;
	harmonic_size.y = _filter_size;
	int2 harmonic_center;
	harmonic_center.x = _filter_size / 2;
	harmonic_center.y = _filter_size / 2;
	harmonic(harmonic_data, 0, _lambda, 0.0, harmonic_center, harmonic_size);
	float* host_harmonic = new float[_filter_size * _filter_size * 2];
	
	cudaMalloc((void**)&_gabor_data, sizeof(float) * _filter_pixels * 2);
	int2 gabor_size;
	gabor_size.x = _filter_size;
	gabor_size.y = _filter_size;
	int2 gabor_center;
	gabor_center.x = _filter_size / 2;
	gabor_center.y = _filter_size / 2;
	multiplyRealComplex(gaussian_data, harmonic_data, _gabor_data, _filter_size * _filter_size);
	float* host_gabor_data = new float[_filter_pixels * 2];
	cudaMemcpy(host_gabor_data,
		_gabor_data,
		sizeof(float) * _filter_pixels * 2,
		cudaMemcpyDeviceToHost);

	//pad the filter
	{
		float* data = host_gabor_data;
		float* target = _filter_image;
		memset(target, 0, sizeof(float) * _padded_pixels * 2);
		int padded_stride = 2 * _padded_size;
		int target_stride = 2 * _target_size;
		for (int i = 0; i < _target_size; ++i)
		{
			memcpy(target, data, sizeof(float) * target_stride);
			target += padded_stride;
			data += target_stride;
		}
	}

	// Copy gabor data into member for texture creation
	_filter_image_mutex.lock();
	memcpy(_host_gabor_data, host_gabor_data, sizeof(float) * _filter_pixels * 2);
	_filter_image_mutex.unlock();
	
	cudaFree(_gabor_data);
	
	cudaMalloc((void**)&_gabor_data, sizeof(float) * _padded_pixels * 2);
	cudaMalloc((void**)&_gpu_image_0, sizeof(float) * _padded_pixels * 2);
	cudaMalloc((void**)&_gpu_image_1, sizeof(float) * _padded_pixels * 2);
	cudaMemcpy(_gabor_data,
		_filter_image,
		sizeof(float) * _padded_pixels * 2,
		cudaMemcpyHostToDevice);

	cufftPlan2d(&_fft_plan, _padded_size, _padded_size, CUFFT_C2C);
	cufftExecC2C(_fft_plan,
		(cufftComplex*)(_gabor_data),
		(cufftComplex*)(_gabor_data),
		CUFFT_FORWARD);
	cudaMemcpy(_filter_image,
		_gabor_data,
		sizeof(float) * _padded_pixels * 2,
		cudaMemcpyDeviceToHost);

	emit newFilterImage();
}

void WorkerThread::createNewFilter()
{
	// Free GPU memory from current filter and CUFFT
	cudaFree(_gabor_data);
	cudaFree(_gpu_image_0);
	cudaFree(_gpu_image_1);
	cufftDestroy(_fft_plan);

	float* gaussian_data;
	cudaMalloc((void**)&gaussian_data, sizeof(float) * _filter_pixels);
	int2 gaussian_size;
	gaussian_size.x = _filter_size;
	gaussian_size.y = _filter_size;
	int2 gaussian_center;
	gaussian_center.x = _filter_size / 2;
	gaussian_center.y = _filter_size / 2;
	gaussian(gaussian_data, _new_theta, _new_sigma, 1.0, gaussian_center, gaussian_size);
	
	float* harmonic_data;
	cudaMalloc((void**)&harmonic_data, sizeof(float) * _filter_pixels * 2);
	int2 harmonic_size;
	harmonic_size.x = _filter_size;
	harmonic_size.y = _filter_size;
	int2 harmonic_center;
	harmonic_center.x = _filter_size / 2;
	harmonic_center.y = _filter_size / 2;
	harmonic(harmonic_data, _new_theta, _new_lambda, _new_psi, harmonic_center, harmonic_size);
	float* host_harmonic = new float[_filter_size * _filter_size * 2];
	
	cudaMalloc((void**)&_gabor_data, sizeof(float) * _filter_pixels * 2);
	int2 gabor_size;
	gabor_size.x = _filter_size;
	gabor_size.y = _filter_size;
	int2 gabor_center;
	gabor_center.x = _filter_size / 2;
	gabor_center.y = _filter_size / 2;
	multiplyRealComplex(gaussian_data, harmonic_data, _gabor_data, _filter_size * _filter_size);
	float* host_gabor_data = new float[_filter_pixels * 2];
	cudaMemcpy(host_gabor_data,
		_gabor_data,
		sizeof(float) * _filter_pixels * 2,
		cudaMemcpyDeviceToHost);

	//pad the filter
	{
		float* data = host_gabor_data;
		float* target = _filter_image;
		memset(target, 0, sizeof(float) * _padded_pixels * 2);
		int padded_stride = 2 * _padded_size;
		int target_stride = 2 * _target_size;
		for (int i = 0; i < _target_size; ++i)
		{
			memcpy(target, data, sizeof(float) * target_stride);
			target += padded_stride;
			data += target_stride;
		}
	}

	// Copy gabor data into member for texture creation
	_filter_image_mutex.lock();
	memcpy(_host_gabor_data, host_gabor_data, sizeof(float) * _filter_pixels * 2);
	_filter_image_mutex.unlock();
	
	cudaFree(_gabor_data);
	
	cudaMalloc((void**)&_gabor_data, sizeof(float) * _padded_pixels * 2);
	cudaMalloc((void**)&_gpu_image_0, sizeof(float) * _padded_pixels * 2);
	cudaMalloc((void**)&_gpu_image_1, sizeof(float) * _padded_pixels * 2);
	cudaMemcpy(_gabor_data,
		_filter_image,
		sizeof(float) * _padded_pixels * 2,
		cudaMemcpyHostToDevice);

	cufftPlan2d(&_fft_plan, _padded_size, _padded_size, CUFFT_C2C);
	cufftExecC2C(_fft_plan,
		(cufftComplex*)(_gabor_data),
		(cufftComplex*)(_gabor_data),
		CUFFT_FORWARD);
	cudaMemcpy(_filter_image,
		_gabor_data,
		sizeof(float) * _padded_pixels * 2,
		cudaMemcpyDeviceToHost);

	// Free temporary GPU memory used for creation of filter
	cudaFree(gaussian_data);
	cudaFree(harmonic_data);

	delete host_harmonic;
	delete host_gabor_data;

	_should_create_new_filter = false;

	emit newFilterImage();
}

void WorkerThread::gaborFilter(float *data)
{
	//get the subimage and pad it
	{
		float* target = _padded_image;
		memset(target, 0, sizeof(float) * _padded_pixels * 2);
		int left = _target_x - _target_size / 2;
		int bottom = _target_y - _target_size / 2;
		data += 2 * left;
		int original_stride = 2 * _original_image_width;
		int padded_stride = 2 * _padded_size;
		int target_stride = 2 * _target_size;
		data += bottom * original_stride;
		for (int i = 0; i < _target_size; ++i)
		{
			memcpy(target, data, sizeof(float) * target_stride);
			target += padded_stride;
			data += original_stride;
		}
	}
	
	float* gpu_image = NULL;
	float* diff_gpu_image = NULL;
	if( _curr_gpu_image == 0 )
	{
		_curr_gpu_image = 1;
		gpu_image = _gpu_image_0;	
		diff_gpu_image = _gpu_image_1;	
	}
	else
	{
		_curr_gpu_image = 0;
		gpu_image = _gpu_image_1;	
		diff_gpu_image = _gpu_image_0;	
	}
	
	cudaMemcpy(gpu_image,
		_padded_image,
		sizeof(float) * _padded_pixels * 2,
		cudaMemcpyHostToDevice);
	
	cufftExecC2C(_fft_plan,
		(cufftComplex*)gpu_image,
		(cufftComplex*)gpu_image,
		CUFFT_FORWARD);
	
	multiplyComplexComplex(_gabor_data, gpu_image, gpu_image, _padded_pixels);
	
	cufftExecC2C(_fft_plan,
		(cufftComplex*)gpu_image,
		(cufftComplex*)gpu_image,
		CUFFT_INVERSE);

	if( _do_difference_images )
	{
		differenceImages( diff_gpu_image, gpu_image, diff_gpu_image, _padded_pixels);
		cudaMemcpy(_padded_image,
			diff_gpu_image,
			sizeof(float) * _padded_pixels * 2,
			cudaMemcpyDeviceToHost);
	}
	else
	{
		cudaMemcpy(_padded_image,
			gpu_image,
			sizeof(float) * _padded_pixels * 2,
			cudaMemcpyDeviceToHost);
	}


	// Extract 128x128 from padded result
	{
		float* data = _padded_image;
		float* target = _result_data;
		int left = _filter_size / 2;
		int bottom = _filter_size / 2;
		data += 2 * left;
		int data_stride = 2 * _padded_size;
		int target_stride = 2 * _target_size;
		data += bottom * data_stride;
		for (int i = 0; i < _target_size; ++i)
		{
			memcpy(target, data, sizeof(float) * target_stride);
			target += target_stride;
			data += data_stride;
		}
	}
}

void WorkerThread::lockImageMutex()
{
	_image_mutex.lock();
}

void WorkerThread::unlockImageMutex()
{
	_image_mutex.unlock();
}

void WorkerThread::wakeWaitingForImageProcessed()
{
	_image_processed.wakeAll();
}

void WorkerThread::lockFilterImageMutex()
{
	_filter_image_mutex.lock();
}

void WorkerThread::unlockFilterImageMutex()
{
	_filter_image_mutex.unlock();
}

float* WorkerThread::getFullImage()
{
	return _full_image;
}

float* WorkerThread::getFilteredImage()
{
	return _result_data;
}

float* WorkerThread::getFilterImage()
{
	return _host_gabor_data;
}

bool WorkerThread::isImageProcessed()
{
	return _is_image_processed;
}

void WorkerThread::setDifferenceImages(bool b)
{
	_difference_images_mutex.lock();
	_do_difference_images = b;	
	_difference_images_mutex.unlock();
}

void WorkerThread::setImageProcessed(bool processed)
{
	_is_image_processed = processed;
}

void WorkerThread::setNewFilterParameters(float theta, float sigma, float lambda, float psi)
{
	_new_filter_mutex.lock();	
	_new_theta = theta;
	_new_sigma = sigma;
	_new_lambda = lambda;
	_new_psi = psi;
	_should_create_new_filter = true;
	_new_filter_mutex.unlock();
}

