#include <cstdio>
#include <cmath>
#include <list>
#include <sys/time.h>
#include <cuda.h>
#include "CUFFT.h"
#include "V4LCamera.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include "Shader.h"
#include "Timer.h"
#include "voConnection.h"
#include "GaborXmlParser.h"
#include <string>
#include <fstream>
#include <list>
#include <time.h>
#include <signal.h>
#include "voInterpreterClient.h"

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
extern "C"
void computeProbabilities(float* a, float* result, int2 divFactor, int2 size, int2 pSize, int2 offset);

const int viewScale = 1;
const int width = 320 * viewScale;
const int height = 240 * viewScale;
const int filterSize = 128 * viewScale;
const int targetSize = 128 * viewScale;
const int filterPixels = filterSize * filterSize;
const int targetPixels = targetSize * targetSize;
int nFilters = 0;
Shader* shader = NULL;
V4LCamera* camera = NULL;
GLuint* filteredTexture;
GLuint* filterTexture;
GLuint targetTexture = 0;
int paddedSize = 0;
int paddedPixels = 0;
int targetX = width / 2;
int targetY = height / 2;
float* paddedImage = NULL;
float** filterImage;
float* gpuPreviousImage = NULL;
float* gpuCurrentImage = NULL;
float* gpuImage = NULL;
float** gaborData;
float* resultData = NULL;
float* targetData = NULL;
float** filteredData;
float** filteredData2;
float** hostGaborData;
float** hostProbs;
float** deviceProbs;
cufftHandle fftPlan;
Timer timer;

const int vo_max_buff_size = 1024;
int2* divFactors;
GaborXmlParser xmlParser;
CompleteConfig config;
voConnection*** connections;
voInterpreterClient *ncstoolsConn;
bool bDifferenceImages = true;
bool differenceFlag = false;
bool bNCSCommunication = true;
bool bPause = false;
int npollct = 0;

// Self-Throttling Timer
#define THROTTLE_TIMER_ID 13
volatile sig_atomic_t throttleTimerFlag = 1;
struct sigevent signalSpec;
timer_t timerID;
struct itimerspec throttleTimerSetting;

// Debug
bool bDebugFiles = false;
ofstream *debugFiles;

void idleCallback()
{
	glutPostRedisplay();
}

void keyboardCallback(unsigned char key, int x, int y)
{
	switch(key)
	{
		case 27: // ESC
			// Close files if needed
			if( bDebugFiles )
			{
				for(int i=0; i<nFilters; ++i)
					debugFiles[i].close();
			}
			exit(0);
		case 32: // SPACE
			bPause = (bPause?false:true);
			// Once paused, 2 new images must be captured
			// before the images should be differenced and
			// a new probability is calculated.
			differenceFlag = false;
		break;
	}
}

void displayCallback()
{
	//TODO
	// read from NCS Tools (if configured to do so)
	// parse / deal with command
	// Given a heartbeat:
	// - increment a counter
	// - if counter limit reached:
	//   --> reset counter
	//   --> set throttleTimerFlag to be 1
	if( config.b_ncstools)
	{
		string tmp;
		// Check if there is anything to read
		if( ncstoolsConn->poll() )
		{
			printf( "Negative polls before positive: %d\n", npollct );
			npollct = 0;
			printf("before getstring.\n");
			tmp = ncstoolsConn->getString();	
			printf("Received: %s\n", tmp.c_str() );
		}
		else
		{
//			printf( "negative poll\n" );
			npollct++;
		}
	}

	// Paused?
	if( bPause )
		return;
	// Throttling
	if( !throttleTimerFlag )
		return;
	throttleTimerFlag = 0;

	// Capture from camera
	camera->capture();

	//get the subimage and pad it
	{
		float* data = camera->frameData();
		float* target = paddedImage;
		float* unpaddedTarget = targetData;
		memset(target, 0, sizeof(float) * paddedPixels * 2);
		int left = targetX - targetSize / 2;
		int bottom = targetY - targetSize / 2;
		data += 2 * left;
		int originalStride = 2 * width;
		int paddedStride = 2 * paddedSize;
		int targetStride = 2 * targetSize;
		data += bottom * originalStride;
		for (int i = 0; i < targetSize; ++i)
		{
			memcpy(target, data, sizeof(float) * targetStride);
			memcpy(unpaddedTarget, data, sizeof(float) * targetStride);
			target += paddedStride;
			data += originalStride;
			unpaddedTarget += targetStride;
		}
	}

	std::swap(gpuCurrentImage, gpuPreviousImage);
	cudaMemcpy(gpuCurrentImage, 
	           paddedImage,
			   sizeof(float) * paddedPixels * 2,
			   cudaMemcpyHostToDevice);

	if( !differenceFlag )
	{
		// Set the difference flag (now, we know that AT LEAST one image has been consecutively
		// aquired without a pause or from the start of the program)
		differenceFlag = true;
		// All that is left to do in this iteration is: rendering and publishing probabilities
		// (both of which we DO NOT want to do until we have two valid images to difference)
		return;
	}

	differenceImages(gpuCurrentImage, gpuPreviousImage, gpuImage, paddedPixels);
	//gpuImage = gpuCurrentImage;

	cufftExecC2C(fftPlan, 
	             (cufftComplex*)gpuImage, 
				 (cufftComplex*)gpuImage, 
				 CUFFT_FORWARD);
#if 1

	int2 pSize; // Padded size
	pSize.x = paddedSize;
	pSize.y = paddedSize;
	int2 offset;
	offset.x = (paddedSize - filterSize)/2;
	offset.y = offset.x;

	for (int filterCt = 0; filterCt < nFilters; ++filterCt)
	{
			float* fData = filteredData[filterCt];
			multiplyComplexComplex(gaborData[filterCt], gpuImage, 
				fData, paddedPixels);
	}

	for (int filterCt = 0; filterCt < nFilters; ++filterCt)
	{
			float* fData = filteredData[filterCt];

			cufftExecC2C(fftPlan, 
			             (cufftComplex*)fData, 
						 (cufftComplex*)fData, 
						 CUFFT_INVERSE);

			// Difference images, if desired, and compute probabilities
			int2 size;
			size.x = targetSize;
			size.y = targetSize;
			if( bDifferenceImages && differenceFlag )
			{
				//differenceImages(fData, 
				//		fData2, 
				//		fData2, 
				//		paddedPixels);

				// Compute probabilities of differenced image
				computeProbabilities(fData, deviceProbs[filterCt],
					divFactors[filterCt], size, pSize, offset );
			}
			else if( differenceFlag )
			{
				// Compute probabilities of gabored imaged (not diffed)
				computeProbabilities(fData, deviceProbs[filterCt],
					divFactors[filterCt], size, pSize, offset );
			}

			if( differenceFlag )
			{
				// Get probabilities from device
				cudaMemcpy(hostProbs[filterCt],
				           deviceProbs[filterCt],
					   sizeof(float) * divFactors[filterCt].x *
						divFactors[filterCt].y,
					   cudaMemcpyDeviceToHost);
			}

			#if 0 // Print probabilities
			{
				int tmp = divFactors[filterCt].x * divFactors[filterCt].y;
				for(int i=0; i<tmp; ++i)
					printf("[%d][%d] - [%d]: %f\n", scale, angle, i, hostProbs[filterCt]);
			}
			#endif
	}

	cufftExecC2C(fftPlan, 
	             (cufftComplex*)gpuImage, 
				 (cufftComplex*)gpuImage, 
				 CUFFT_INVERSE);
#endif

#if 1 // Render
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	for (int filterCt = 0; filterCt < nFilters && filterCt < 12; ++filterCt)
	{
		float *fData = filteredData[filterCt];
				
		cudaMemcpy(paddedImage,
		           fData,
				   sizeof(float) * paddedPixels * 2,
				   cudaMemcpyDeviceToHost);
		{
			float* data = paddedImage;
			float* target = resultData;
			int left = filterSize / 2;
			int bottom = filterSize / 2;
			data += 2 * left;
			int dataStride = 2 * paddedSize;
			int targetStride = 2 * targetSize;
			data += bottom * dataStride;
			for (int i = 0; i < targetSize; ++i)
			{
				memcpy(target, data, sizeof(float) * targetStride);
				target += targetStride;
				data += dataStride;
			}
		}

		if( filterCt < 4 )
		{
			glViewport(filterCt * targetSize / viewScale, 2 * targetSize / viewScale, targetSize / viewScale, targetSize / viewScale);
		}
		else if( filterCt < 8 )
		{
			glViewport( (filterCt % 4) * targetSize / viewScale, 1 * targetSize / viewScale, targetSize / viewScale, targetSize / viewScale);
		}
		else // < 12
		{
			glViewport( (filterCt % 4) * targetSize / viewScale, 0, targetSize / viewScale, targetSize / viewScale);
		}
		glActiveTexture(GL_TEXTURE0);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, targetTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D,
		             0,
					 GL_LUMINANCE_ALPHA32F_ARB,
					 targetSize,
					 targetSize,
					 0,
					 GL_LUMINANCE_ALPHA,
					 GL_FLOAT,
					 resultData);
		shader->begin();
		shader->set("filteredTexture", 0);
		//shader->set("scale", 1.0f / 255.0f / (float)paddedPixels);
		//shader->set("scale", 1.0f / (float)paddedPixels);
		shader->set("scale", 1.0f / 256.0f / 256.0f);
		//shader->set("scale", 1.0f / 256.0f / 256.0f / 256.0f);
		//shader->set("scale", 1.0f);
		glBegin(GL_QUADS);
			glTexCoord2f(0,1); glVertex4f(-1,-1,0,1);
			glTexCoord2f(1,1); glVertex4f( 1,-1,0,1);
			glTexCoord2f(1,0); glVertex4f( 1, 1,0,1);
			glTexCoord2f(0,0); glVertex4f(-1, 1,0,1);
		glEnd();
		shader->end();
	}

	glActiveTexture(GL_TEXTURE0);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, targetTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D,
	             0,
				 GL_LUMINANCE_ALPHA32F_ARB,
				 targetSize,
				 targetSize,
				 0,
				 GL_LUMINANCE_ALPHA,
				 GL_FLOAT,
				 targetData);
	glViewport(4 * targetSize / viewScale, 0, targetSize * 3 / viewScale, targetSize * 3 / viewScale);
	shader->begin();
	shader->set("filteredTexture", 0);
	shader->set("scale", 1.0f / 255.0f);
	glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
	glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
	glBegin(GL_QUADS);
		glTexCoord2f(0,1); glVertex4f(-1,-1,0,1);
		glTexCoord2f(1,1); glVertex4f( 1,-1,0,1);
		glTexCoord2f(1,0); glVertex4f( 1, 1,0,1);
		glTexCoord2f(0,0); glVertex4f(-1, 1,0,1);
	glEnd();
	shader->end();

	// Only render first 12 filter textures
	for (int filterCt = 0; filterCt < nFilters && filterCt < 12; ++filterCt)
	{
			glBindTexture(GL_TEXTURE_2D, filterTexture[filterCt]);

			if( filterCt < 4 )
				glViewport(filterCt * targetSize / viewScale, 5 * targetSize / viewScale, targetSize / viewScale, targetSize / viewScale);
			else if( filterCt < 8 )
				glViewport( (filterCt % 4) * targetSize / viewScale, 4 * targetSize / viewScale, targetSize / viewScale, targetSize / viewScale);
			else // < 12
				glViewport( (filterCt % 4) * targetSize / viewScale, 3 * targetSize / viewScale, targetSize / viewScale, targetSize / viewScale);

			shader->begin();
			shader->set("filteredTexture", 0);
			shader->set("scale", 255.0f);
			glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
			glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
			glBegin(GL_QUADS);
				glTexCoord2f(0,1); glVertex4f(-1,-1,0,1);
				glTexCoord2f(1,1); glVertex4f( 1,-1,0,1);
				glTexCoord2f(1,0); glVertex4f( 1, 1,0,1);
				glTexCoord2f(0,0); glVertex4f(-1, 1,0,1);
			glEnd();
			shader->end();
	}
#endif // if render

	// Publish the new probabilities to NCS
	if( bNCSCommunication && differenceFlag )
	{
		int divisions, i, j;
		for(i=0; i<nFilters; ++i)
		{
			divisions = divFactors[i].x * divFactors[i].y;
			for(j=0; j<divisions; ++j)
			{
				connections[i][j]->publish( (char*)&hostProbs[i][j], sizeof(float), 1 );
			}
		}
	}

	// Write probabilities to file
	if( bDebugFiles && differenceFlag )
	{
		int i, j, divisions;
		for(i=0; i<nFilters; ++i)
		{
			divisions = divFactors[i].x * divFactors[i].y;
			for(j=0; j<divisions-1; ++j)
			{
				debugFiles[i] << hostProbs[i][j] << "\t";
			}
			debugFiles[i] << hostProbs[i][divisions-1] << endl;
		}
	}

	glutSwapBuffers();
	char fps[512];
	sprintf(fps, "FPS: %4.2f", 1.0 / timer.stop());
	glutSetWindowTitle(fps);
	timer.start();
}

void createVoConnections()
{
	int i,j;
	int divisions;
	char tmp[1024];
	// Create voConnections for each filter
	connections = new voConnection**[ config.filters.size() ];
	for(i=0; i<config.filters.size(); ++i)
	{
		// Create a voConnection for each region based on
		// division factors
		divisions = config.filters[i].div_x * config.filters[i].div_y;
		connections[i] = new voConnection*[ divisions ];
		for(j=0; j<divisions; ++j)
		{
			Connection_Info conn_info;
			sprintf( tmp, "%d", config.filters[i].port );
			conn_info.port = string( tmp );
			conn_info.host = config.filters[i].host;
			conn_info.type = BINARY_PUBLISH;
			// Create the connection name:
			// [type][job][name][region]
			sprintf( tmp, "%s%s%s%d", 
				config.filters[i].type.c_str(), 
				config.filters[i].job.c_str(),
				config.filters[i].name.c_str(), 
				j+1 );
			conn_info.connect_name = string( tmp );
			// Save the connection
			connections[i][j] = new voConnection( conn_info, vo_max_buff_size );
		}
	}
}

bool connectVoConnections()
{
	int i,j;
	int divisions;
	for(i=0; i<config.filters.size(); ++i)
	{
		divisions = config.filters[i].div_x * config.filters[i].div_y;
		for(j=0; j<divisions; ++j)
		{
			if( !connections[i][j]->voConnect() )
				return false;
		}
	}
	return true;
}

void throttleTimerHandler(int sig)
{
	throttleTimerFlag = 1;	
}

using namespace std;

int main(int argc, char** argv) {

	if( argc < 3 )
	{
		printf("Usage: gabor <video source> <config.xml> [-noNCS] [-writetofile]\n");
		return 0;
	}

	// Load the XML file
	string filename(argv[2]);
	string error;
	config = xmlParser.loadConfig(filename, error);
	if( !config.isValid() )
	{
		printf("Error: %s\n", error.c_str() );	
		return 0;
	}
	printf("XML Load successful: %d filters\n", (int)config.filters.size());
	nFilters = (int)config.filters.size();

	// Check if there are extra (optional) parameters
	for(int i=argc-1; i>2; --i)
	{
		string arg(argv[i]);
		if( arg == string("-writetofile") )
		{
			bDebugFiles = true;	
			printf( "Write to file mode.\n" );
		}
		else if( arg == string("-noNCS") )
		{
			bNCSCommunication = false;
			printf( "No NCS connections mode.\n" );
		}
		else
		{
			printf("Usage: gabor <video source> <config.xml> [-nonet] [-writetofile]\n");
			return 0;
		}
	}

	// Video source
	string videoSource( argv[1] );

	// Difference images?
	bDifferenceImages = config.b_diff_images;

	// Debug setup
	if( bDebugFiles )
	{
		char filename[100];
		string basename = "probabilities_f";
		debugFiles = new ofstream[nFilters];
		for(int i=0; i<nFilters; ++i)
		{
			sprintf( filename, "%s%d.txt", basename.c_str(), i );
			debugFiles[i].open( filename, ofstream::trunc );
			debugFiles[i].setf( ofstream::fixed, ofstream::floatfield );
			debugFiles[i].precision( 5 );
		}
	}

	// NCS Communication setup
	if( bNCSCommunication )
	{
		createVoConnections();
		if( !connectVoConnections() )
		{
			printf( "Error: voConnect failed\n" );
			return 0;
		}
		else
		{
			printf( "NCS connections established successfully\n" );
		}
	}

	// NCSTools connection setup
	if( config.b_ncstools )
	{
		char tmp[20];
		Connection_Info conn_info;
		//sprintf( tmp, "%d", config.port_ncstools );
		sprintf( tmp, "%d", 20001 );
		conn_info.port = string( tmp );
		conn_info.host = string( "localhost" );
		//conn_info.host = config.host_ncstools;
		conn_info.type = ASCII_SUBSCRIBE;
		// Create the connection name:
		conn_info.connect_name = string( "voInterpreterClientGabor" );
		// create the connection
		//ncstoolsConn = new voInterpreterClient( conn_info, vo_max_buff_size );
		ncstoolsConn = new voInterpreterClient( conn_info, 4096 );

		// Attempt to establish the connection
		if( !ncstoolsConn->voConnect() )
		{
			printf( "Could not establish connection with NCSTools.\n" );
			return 0;
		}
		printf( "NCSTools connection successful.\n" );
	}

	// Self-throttling setup
	if( !config.b_server_throttle || !config.b_ncstools )
	{	
		int err;
		// setup the signal handler
		signal( SIGUSR1, throttleTimerHandler );
		
		// setup the timer specification
		signalSpec.sigev_notify = SIGEV_SIGNAL;
		signalSpec.sigev_signo = SIGUSR1;
		signalSpec.sigev_value.sival_int = THROTTLE_TIMER_ID;
		
		// create the timer
		err = timer_create(CLOCK_REALTIME, &signalSpec, &timerID);
		if( err < 0 )
		{
			printf( "Error creating self-throttling timer\n" );
			return 0;
		}
	
		// Set the timer (and start the timer)
		double frequency = 1000.0 / config.send_freq; // milliseconds
		double nanoseconds = 1.0e9 / frequency;
		double seconds = 0.0;
		if( nanoseconds > 1.0e9 )
		{
			seconds = nanoseconds / 1.0e9;
			nanoseconds = nanoseconds - seconds*1.0e9;
		}
        throttleTimerSetting.it_value.tv_sec = seconds;
        throttleTimerSetting.it_value.tv_nsec = nanoseconds;
        throttleTimerSetting.it_interval.tv_sec = seconds;
        throttleTimerSetting.it_interval.tv_nsec = nanoseconds;
        err = timer_settime( timerID, 0, &throttleTimerSetting, NULL );
		if( err < 0 )
		{
			printf( "Error setting throttling timer\n" );
			return 0;
		}
		printf( "Successfully started throttling timer\n" );
	}

	divFactors = new int2[nFilters];
	filterImage = new float*[nFilters];
	gaborData = new float*[nFilters];
	filteredData = new float*[nFilters];
	filteredData2 = new float*[nFilters];
	hostGaborData = new float*[nFilters];
	hostProbs = new float*[nFilters];
	deviceProbs = new float*[nFilters];

	// Set division factors for each filter
	for(int i=0; i<nFilters; ++i)
	{
		divFactors[i].x = config.filters[i].div_x;
		divFactors[i].y = config.filters[i].div_y;
	}

	// Allocate memory for probabilities for each filter
	int aSize;
	for(int i=0; i<nFilters; ++i)
	{	
		aSize = divFactors[i].x * divFactors[i].y;
		hostProbs[i] = new float[ aSize ];
		cudaMalloc((void**)&deviceProbs[i], sizeof(float)*aSize);
	}

	// Calculate padded size
	paddedSize = filterSize + targetSize;
	int log = 0;
	while (paddedSize != 1)
	{
		paddedSize >>= 1;
		++log;
	}
	paddedSize = 1 << log;
	if (paddedSize < filterSize + targetSize)
		paddedSize <<= 1;
	paddedPixels = paddedSize * paddedSize;
	paddedImage = new float[paddedPixels * 2];
	for (int i = 0; i < nFilters; ++i)
		filterImage[i] = new float[paddedPixels * 2];
	resultData = new float[targetPixels * 2];
	targetData = new float[targetPixels * 2];

	// GLUT setup
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(targetSize * 4 * 2 / viewScale, targetSize * 3 * 2 / viewScale);
	glutCreateWindow("Gabor Filter");
	glutDisplayFunc(displayCallback);
	glutKeyboardFunc(keyboardCallback);
	glutIdleFunc(idleCallback);
	CUFFT::init();
	glewInit();

	// Shader setup
	shader = new Shader();
	shader->addShader("render.vert", GL_VERTEX_SHADER);
	shader->addShader("render.frag", GL_FRAGMENT_SHADER);
	shader->compile();

	filteredTexture = new GLuint[ (nFilters<13?nFilters:12) ];
	filterTexture = new GLuint[ (nFilters<13?nFilters:12) ];

	glGenTextures( (nFilters<13?nFilters:12), filteredTexture);
	glGenTextures( (nFilters<13?nFilters:12), filterTexture);
	glGenTextures(1, &targetTexture);

	// Create the camera object
	camera = new V4LCamera(videoSource.c_str(), width, height);

	cufftPlan2d(&fftPlan, paddedSize, paddedSize, CUFFT_C2C);

	// Create gabor filters
	for (int filterCt = 0; filterCt < nFilters; ++filterCt)
	{
		float* gaussianData;
		cudaMalloc((void**)&gaussianData, sizeof(float) * filterPixels);
		int2 gaussianSize; gaussianSize.x = filterSize; gaussianSize.y = filterSize;
		int2 gaussianCenter; gaussianCenter.x = filterSize / 2; gaussianCenter.y = filterSize / 2;
		gaussian(gaussianData, config.filters[filterCt].theta, config.filters[filterCt].sigma, 1.0, gaussianCenter, gaussianSize);

		float* harmonicData;
		cudaMalloc((void**)&harmonicData, sizeof(float) * filterPixels * 2);
		int2 harmonicSize; harmonicSize.x = filterSize; harmonicSize.y = filterSize;
		int2 harmonicCenter; harmonicCenter.x = filterSize / 2; harmonicCenter.y = filterSize / 2;
		harmonic(harmonicData, config.filters[filterCt].theta, config.filters[filterCt].lambda, config.filters[filterCt].psi, harmonicCenter, harmonicSize);
		float* hostHarmonic = new float[filterSize * filterSize * 2];

		cudaMalloc((void**)&(gaborData[filterCt]), sizeof(float) * filterPixels * 2);
		int2 gaborSize; 
		gaborSize.x = filterSize; 
		gaborSize.y = filterSize;
		int2 gaborCenter; 
		gaborCenter.x = filterSize / 2; 
		gaborCenter.y = filterSize / 2;
		multiplyRealComplex(gaussianData, harmonicData, gaborData[filterCt], filterSize * filterSize);
		hostGaborData[filterCt] = new float[filterPixels * 2];
		cudaMemcpy(hostGaborData[filterCt],
		           gaborData[filterCt],
				   sizeof(float) * filterPixels * 2,
				   cudaMemcpyDeviceToHost);
	
		// Pad the filter
		{
			float* data = hostGaborData[filterCt];
			float* target = filterImage[filterCt];
			memset(target, 0, sizeof(float) * paddedPixels * 2);
			int paddedStride = 2 * paddedSize;
			int filterStride = 2 * filterSize;
			for (int i = 0; i < filterSize; ++i)
			{
				memcpy(target, data, sizeof(float) * filterStride);
				target += paddedStride;
				data += filterStride;
			}
		}

		// Generate filter textures
		if( filterCt < 12 )
		{
			glEnable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D, filterTexture[filterCt]);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexImage2D(GL_TEXTURE_2D,
			             0,
						 GL_LUMINANCE_ALPHA32F_ARB,
						 filterSize,
						 filterSize,
						 0,
						 GL_LUMINANCE_ALPHA,
						 GL_FLOAT,
						 hostGaborData[filterCt]);
		}

		cudaFree(gaussianData);
		cudaFree(harmonicData);
		cudaFree(gaborData[filterCt]);
		cudaMalloc((void**)&(gaborData[filterCt]), sizeof(float) * paddedPixels * 2);
		cudaMemcpy(gaborData[filterCt],
		           filterImage[filterCt],
				   sizeof(float) * paddedPixels * 2,
				   cudaMemcpyHostToDevice);
		cufftExecC2C(fftPlan, 
		             (cufftComplex*)(gaborData[filterCt]), 
					 (cufftComplex*)(gaborData[filterCt]), 
					 CUFFT_FORWARD);
		cudaMalloc((void**)&(filteredData[filterCt]), sizeof(float) * paddedPixels * 2);
		cudaMalloc((void**)&(filteredData2[filterCt]), sizeof(float) * paddedPixels * 2);
	}
	cudaMalloc((void**)&gpuPreviousImage, sizeof(float) * paddedPixels * 2);
	cudaMalloc((void**)&gpuCurrentImage, sizeof(float) * paddedPixels * 2);
	cudaMalloc((void**)&gpuImage, sizeof(float) * paddedPixels * 2);
	timer.start();
	glutMainLoop();

    return 0;
}
