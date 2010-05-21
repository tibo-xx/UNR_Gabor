#include "Shader.h"
#include <QWidget>
#include <QApplication>
#include <QGLWidget>
#include <GL/gl.h>
#include <GL/glu.h>
#include "GLContext.h"
#include <QDebug>

GLuint texture;

// Default constructor
GLContext :: GLContext( QString video_source, QWidget *parent ) 
				: QGLWidget(parent)
{
	_initial_render = false;
	_is_filter_image_valid = false;

	_worker_thread = new WorkerThread(video_source, this);
	
	_original_image_width = 320;
	_original_image_height = 240;

	_filter_opacity = 1.0;

	// FPS Display
	_fps_qtimer = new QTimer(this);
	_fps_timer.start();
	connect( _fps_qtimer, SIGNAL( timeout() ), this, SLOT( updateFPS() ) );
	_fps_qtimer->setSingleShot(false);
	_fps_qtimer->start(1750);
	_frame_ct = 0;

	connect(_worker_thread, SIGNAL(filterComplete()), this, SLOT( displayImages()));
	connect(_worker_thread, SIGNAL(newFilterImage()), this, SLOT( updateFilterImage()));
}   

// Default destructor
GLContext :: ~GLContext()
{               
}               

// Perform necessary cleanup operations
void GLContext::cleanup()
{
	_worker_thread->cleanup();
	_worker_thread->wait();
}

// Start worker thread for camera capture and filtering
void GLContext::start()
{
	_worker_thread->start();
}


void GLContext::updateFPS()
{
	float time = (float)_fps_timer.stop() / 1000000.0;	
	emit newFPS( _frame_ct / time );
	_frame_ct = 0;
	_fps_timer.start();
}

// Retrieve full image and respective filtered image from worker thread and diaplay.
void GLContext::displayImages()
{
	_frame_ct++;
	_worker_thread->lockImageMutex();
	if( !_worker_thread->isImageProcessed() )
	{
		// Create texture for full image
		float *full_image = _worker_thread->getFullImage();
		glBindTexture(GL_TEXTURE_2D, _texture[0]);
		glTexImage2D(GL_TEXTURE_2D,
			0,
			GL_LUMINANCE_ALPHA32F_ARB,
			_original_image_width,
			_original_image_height,
			0,
			GL_LUMINANCE_ALPHA,
			GL_FLOAT,
			full_image);
		// Create texture for fitlered image
		int filtered_size = 128;
		float *filtered_image = _worker_thread->getFilteredImage();
		glBindTexture(GL_TEXTURE_2D, _texture[1]);
		glTexImage2D(GL_TEXTURE_2D,
			0,
			GL_LUMINANCE_ALPHA32F_ARB,
			filtered_size,
			filtered_size,
			0,
			GL_LUMINANCE_ALPHA,
			GL_FLOAT,
			filtered_image);

		_worker_thread->setImageProcessed(true);
		_initial_render = true;
	}
	_worker_thread->wakeWaitingForImageProcessed();
	_worker_thread->unlockImageMutex();

	updateGL();
}

// Retrieve filter image from worker thread and update texture for display.
void GLContext :: updateFilterImage()
{
	_worker_thread->lockFilterImageMutex();
	int target_size = 128;
	float* image = _worker_thread->getFilterImage();
	glBindTexture(GL_TEXTURE_2D, _texture[2]);
	glTexImage2D(GL_TEXTURE_2D,
		0,
		GL_LUMINANCE_ALPHA32F_ARB,
		target_size,
		target_size,
		0,
		GL_LUMINANCE_ALPHA,
		GL_FLOAT,
		image);
	_worker_thread->unlockFilterImageMutex();
	_is_filter_image_valid = true;
}

// OpenGL initialization
void GLContext :: initializeGL()
{
	makeCurrent ();

	glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );

	// Setup the world coordinate system
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(-1.0, 1.0, -1.0, 1.0);

	// Blending (filter opacity)
	glEnable( GL_BLEND );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

	// Textures
	glEnable (GL_TEXTURE_2D);
	glGenTextures (3, _texture);
	glBindTexture(GL_TEXTURE_2D, _texture[0]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, _texture[1]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, _texture[2]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glewInit();
	_shader = new Shader();
	_shader->addShader("render.vert", GL_VERTEX_SHADER);
	_shader->addShader("render.frag", GL_FRAGMENT_SHADER);
	_shader->compile();

	_worker_thread->start();
}

// Called when widget is resized
void GLContext :: resizeGL( int width, int height )
{
	// Set the viewport to be the full size of the canvas
	glViewport(0, 0, width, height);
}

// OpenGL rendering
void GLContext :: paintGL()
{
	makeCurrent();

	glClear(GL_COLOR_BUFFER_BIT);

	if( !_initial_render || !_is_filter_image_valid)
	{
		return;
	}

	glEnable( GL_TEXTURE_2D );

	// Render original image
	glBindTexture(GL_TEXTURE_2D, _texture[0]);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	_shader->begin();
	_shader->set("filteredTexture", 0);
	_shader->set("scale", 1.0f / 255.0f);
	_shader->set("opacity", 1.0f);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 1.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, -1.0f);
	glEnd();
	_shader->end();

	// Render the filtered image
	glPushAttrib( GL_ALL_ATTRIB_BITS );
	float y_offset = 128.0f/240.0f;
	float x_offset = 128.0f/320.0f;
	glBindTexture(GL_TEXTURE_2D, _texture[1]);
	_shader->begin();
	_shader->set("filteredTexture", 0);
	_shader->set("scale", 1.0f / 255.0f / (float) (256.0f) );
	_shader->set("opacity", _filter_opacity);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(-x_offset, -y_offset);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(-x_offset, y_offset);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(x_offset, y_offset);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(x_offset, -y_offset);
	glEnd();
	_shader->end();
	glPopAttrib();

	// Render the gabor filter
	float size_x = 2.0f / 5.0f;
	float size_y = size_x * 1.33333333f;
	glBindTexture(GL_TEXTURE_2D, _texture[2]);
	_shader->begin();
	_shader->set("filteredTexture", 0);
	_shader->set("scale", 255.0f);
	_shader->set("opacity", 1.0f);
	glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
	glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
	glBegin(GL_QUADS);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(-1.0f, 1.0f-size_y);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f+size_x, 1.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f+size_x, 1.0f-size_y);
	glEnd();
	_shader->end();
}

// Set new filter parameters.
void GLContext::setNewFilterParameters(float theta, float sigma, float lambda, float psi)
{
	_worker_thread->setNewFilterParameters(theta, sigma, lambda, psi);
}

// Set opacity of filter overlay for rendering
void GLContext::setFilterOpacity(float opacity)
{
	_filter_opacity = opacity;
}

