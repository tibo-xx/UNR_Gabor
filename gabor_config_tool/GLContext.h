
/*
* File name: GLContext.h
*/

#ifndef __GLCONTEXT_H__
#define __GLCONTEXT_H__

#include "Shader.h"

#include <QWidget>
#include <QTimer>
#include <GL/gl.h>
#include <QGLWidget>
#include <QPoint>
#include <QString>

#include "WorkerThread.h"
#include "Timer.h"

/**
* Widget for OpenGL context (for display of camera video and filter)
*/
class GLContext : public QGLWidget
{

	Q_OBJECT 

public:

	/**
	* Default constructor.
	*/
	GLContext(QString video_source, QWidget *parent=0 );

	/**
	* Default destructor.
	*/
	~GLContext();

	/**
	* Perform necessary cleanup operations on program exit.
	*/
	void cleanup();

signals:

	/**
	* Signal to emit when a new FPS value has been calculated.
	*/
	void newFPS( float );

public slots:

	/**
	* Retrieve full image and respective filtered image from worker thread and display.
	*/
	void displayImages();

	/**
	*  Start worker thread for camera capture and filtering.
	*/
	void start();

	/**
	* Set new filter parameters.
	* @param theta Theta parameter for gabor filter.
	* @param sigma Sigma parameter for gabor filter.
	* @param lambda Lambda parameter for gabor fileter.
	* @param psi Psi parameter for gabor fileter.
	*/
	void setNewFilterParameters(float theta, float sigma, float lambda, float psi);

	/**
	* Set new opacity for filter image (overlay on raw camera image)
	*/
	void setFilterOpacity(float);

	/**
	* Set status of image differencing.
	* @param b True = difference images / False = do NOT difference images.
	*/
	void setDifferenceImages(int b)
	{ 
		if( b ) 
			_worker_thread->setDifferenceImages(true);
		else 
			_worker_thread->setDifferenceImages(false);
	}

private slots:

	/**
	* Calculate new FPS and emit signal.
	*/
	void updateFPS();

	/**
	* Retrieve filter image from worker thread and update display.
	*/
	void updateFilterImage();

protected:

	/**
	* OpenGL initialization.
	*/
	void initializeGL();

	/**
	* Executed when this widget is resized.
 	* @param w Width of new size.
 	* @param h Height of new size.
	*/
	void resizeGL( int w, int h );

	/**
	* Perform all rendering. Should not directly call this function,
	* but instead use updateGL() to trigger a render in the Qt event queue.
	*/
	void paintGL();

private:

	WorkerThread *_worker_thread; // Worker thread to do camera capture and filtering
	float _filter_opacity; // Opacity of filter overlay
	int _original_image_width; // Width of image received from camera
	int _original_image_height; // Height of image received from camera
	bool _initial_render; // Flag for initial render
	bool _is_filter_image_valid; // Flag for validity of filter image
	GLuint _texture[3];

	Shader *_shader; // To perform simple normalizatino on filtered image data for rendering

	QTimer *_fps_qtimer;
	Timer _fps_timer;
	int _frame_ct;

};

#endif // __GLCONTEXT_H__

