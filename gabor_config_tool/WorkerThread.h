/*
* File name: WorkerThread.h
*/

#ifndef __WORKER_THREAD__
#define __WORKER_THREAD__

#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QString>
#include <cuda_runtime.h>
#include <cufft.h>
#include "V4LCamera.h"
#include <time.h>
#include <signal.h>

/**
* Thread to capture from camera and filter.
*/
class WorkerThread : public QThread {

	Q_OBJECT

public:

	/**
	* Default constructor.
	*/
	WorkerThread(QString video_source, QObject*);

	/**
	* Default destructor.
	*/
	~WorkerThread();
	
	/**
	* Perform all necessary cleanup / terminate thread.
	*/
	void cleanup();

	/**
	* Start thread.
	*/
	void run();	

	/**
	* Lock mutex for camera image.
	*/
	void lockImageMutex();

	/**
	* Unlock mutex for camera image.
	*/
	void unlockImageMutex();

	/**
	* Wake threads waiting for image to be processed.
	*/
	void wakeWaitingForImageProcessed();

	/**
	* Lock mutex for filter image.
	*/
	void lockFilterImageMutex();

	/**
	* Unlock mutex for filter image.
	*/
	void unlockFilterImageMutex();

	/**
	* Return full image data.
	*/
	float* getFullImage();

	/**
	* Return filetered image data.
	*/
	float* getFilteredImage();

	/**
	* Return image data of filter.
	*/
	float* getFilterImage();

	/**
	* Return if image has been processed.
	*/
	bool isImageProcessed();

	/**
	* Update status of image-processed flag.
	*/
	void setImageProcessed(bool);

public slots:

	/**
	* Set new filtere parameters.
	* @param theta Theta parameter of gabor filter.
	* @param sigma Sigma parameter of gabor filter.
	* @param lambda Lambda parameter of gabor filter.
	* @param psi Psi parameter of gabor filter.
	*/
	void setNewFilterParameters(float theta, float sigma, float lambda, float psi);

	/**
	* Set difference images flag
	* @param b True = difference images / False = do NOT difference images
	*/
	void setDifferenceImages(bool b);

signals:

	/**
	* To be emitted when the filter is complete
	*/
	void filterComplete();

	/**
	* To be emitted when a new filter image has been created (for display purposes).
	*/
	void newFilterImage();

private:

	/**
	* Create initial gabor filter.
	*/
	void createInitialFilter();

	/**
	* Create a new filter based on parameters set with the setNewFilterParameters function.
	*/
	void createNewFilter();

	/**
	* Perform gabor filter
	* @param data Data to be filtered.
	*/
	void gaborFilter(float *data);

	QMutex _image_mutex; // Image mutex
	QWaitCondition _image_processed; // Wait condition for image processing
	bool _is_image_processed; // Flag for image processing
	V4LCamera *_camera; // Camera object
	int _original_image_width; // Full image width (from camera)
	int _original_image_height; // Full image height
	float _full_image[320*240*2];
	static const int _filter_size = 128; // Filter size is 128x128
	static const int _target_size = 128;
	static const int _filter_pixels = _filter_size * _filter_size; // Pixel count of filter
	static const int _target_pixels = _target_size * _target_size;
	int _padded_size; // Size of padded image
	int _padded_pixels; // Count of padded pixels
	int _target_x; // Center of filter within full image
	int _target_y;
	float _sigma;
	float _lambda;
	float* _result_data; // Filtered data
	float* _padded_image; // Padded image data
	float* _filter_image; // Image of filter
	float* _gpu_image_0; // Need to GPU images for differencing
	float* _gpu_image_1;
	int _curr_gpu_image; // Current image (for differencing)
	float* _gabor_data; // Data of gabor filter (device)
	float* _host_gabor_data; // Data of gabor filter (host)
	cufftHandle _fft_plan; // Handle for CUDA FFT object
	
	float _new_theta;
	float _new_sigma;
	float _new_lambda;
	float _new_psi;
	bool _should_create_new_filter;

	QMutex _new_filter_mutex;
	QMutex _filter_image_mutex;
	QMutex _cleanup_mutex;
	QMutex _difference_images_mutex;

	bool _should_terminate; // For thread termination via external widget (GLContext)
	bool _do_difference_images;

};

#endif
