#include "V4LCamera.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>
#include <fcntl.h>
#include <malloc.h>
#include <string.h>

V4LCamera::V4LCamera(const char* devicePath, int width, int height)
{
	struct stat deviceStatus;
	if (-1 == stat(devicePath, &deviceStatus))
	{
		printf("Error: Could not stat device %s\n", devicePath);
		exit(-1);
	}
	if (!S_ISCHR(deviceStatus.st_mode))
	{
		printf("Error: %s is not a character device\n", devicePath);
		exit(-1);
	}
	_file = open(devicePath, O_RDWR | O_NONBLOCK, 0);
	if (-1 == _file)
	{
		printf("Error: Could not open file %s\n", devicePath);
		exit(-1);
	}
	if (-1 == _command(VIDIOC_QUERYCAP, &_capabilities))
	{
		printf("Error: %s is not a V4L2 device", devicePath);
		exit(-1);
	}
	if (!(_capabilities.capabilities & V4L2_CAP_VIDEO_CAPTURE))
	{
		printf("Error: %s is not a capture device", devicePath);
		exit(-1);
	}
	struct v4l2_cropcap cropCapabilities;
	cropCapabilities.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (-1 == _command(VIDIOC_CROPCAP, &cropCapabilities))
	{
		struct v4l2_crop cropping;
		cropping.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		cropping.c = cropCapabilities.defrect;
		_command(VIDIOC_S_CROP, &cropping);
	}
	struct v4l2_format format;
	format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
	format.fmt.pix.field = V4L2_FIELD_INTERLACED;
	format.fmt.pix.width = width;
	format.fmt.pix.height = height;
	if (-1 == _command(VIDIOC_S_FMT, &format))
	{
		printf("Error: Could not set format\n");
		exit(-1);
	}
	struct v4l2_input channel;
	channel.index = 0;
	if (-1 == _command(VIDIOC_S_INPUT, &channel))
	{
		printf("Error: Could not set channel\n");
		exit(-1);
	}
	struct v4l2_requestbuffers request;
	memset(&request, 0, sizeof(request));
	request.count = numBuffers;
	request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	request.memory = V4L2_MEMORY_MMAP;
	if (-1 == _command(VIDIOC_REQBUFS, &request))
	{
		printf("Error: Could not request memory mapping\n");
		exit(-1);
	}

	for (int i = 0; i < numBuffers; ++i)
	{
		struct v4l2_buffer buffer;
		memset(&buffer, 0, sizeof(buffer));
		buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buffer.memory = V4L2_MEMORY_MMAP;
		buffer.index = 0;
		if (-1 == _command(VIDIOC_QUERYBUF, &buffer))
		{
			printf("Error: Could not get MMAP information for buffer %d\n", i);
			exit(-1);
		}
		_buffers[i] = (unsigned char*)mmap(NULL, 
		                                   buffer.length,
						                   PROT_READ | PROT_WRITE,
						                   MAP_SHARED,
						                   _file,
						                   buffer.m.offset);
		if (MAP_FAILED == _buffers[i])
		{
			printf("Error: MMAP failed for buffer %d\n", i);
			exit(-1);
		}
		memset(&buffer, 0, sizeof(buffer));
		buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buffer.memory = V4L2_MEMORY_MMAP;
		buffer.index = i;
		if (-1 == _command(VIDIOC_QBUF, &buffer))
		{
			printf("Error: Could not queue buffer\n");
			exit(-1);
		}
	}
	int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (-1 == _command(VIDIOC_STREAMON, &type))
	{
		printf("Error: Could not start streaming\n");
		exit(-1);
	}
	_frameData = new float[width * height * 2];
}

void V4LCamera::capture()
{
	struct v4l2_buffer buffer;
	buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buffer.memory = V4L2_MEMORY_MMAP;
	int result = 0;
	while(-1 == (result = _command(VIDIOC_DQBUF, &buffer)) && 
	      EAGAIN == errno);
	if (-1 == result)
	{
		printf("Error: Could not capture frame\n");
		exit(-1);
	}
	unsigned char* yuyv = _buffers[buffer.index];
	unsigned char* end = yuyv + buffer.length;
	float* y = _frameData;
	for (; yuyv != end; y += 4, yuyv += 4)
	{
		y[0] = yuyv[0];
		y[1] = 0.0f;
		y[2] = yuyv[2];
		y[3] = 0.0f;
	}
	if (-1 == _command(VIDIOC_QBUF, &buffer))
	{
		printf("Error: Could not re-queue buffer\n");
		exit(-1);
	}

}

float* V4LCamera::frameData()
{
	return _frameData;
}

int V4LCamera::_command(int command, void* arg)
{
	int result = -1;
	while (-1 == (result = ioctl(_file, command, arg)) && EINTR == errno);
	return result;
}

