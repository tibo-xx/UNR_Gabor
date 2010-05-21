#pragma once
#include <linux/videodev2.h>

class V4LCamera
{
public:
	static const int numBuffers = 1;
	V4LCamera(const char* devicePath, int width, int height);
	void capture();
	float* frameData();
private:
	int _file;
	int _command(int control, void* arg);
	struct v4l2_capability _capabilities;
	unsigned char* _buffers[numBuffers];
	float* _frameData;
};
