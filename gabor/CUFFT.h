#pragma once

class CUFFT
{
public:
	static void init();
	static void fft(float* data, int cols, int rows, bool forward);
private:
	CUFFT();
	~CUFFT();
	static CUFFT* _singleton;

	void _init();
	void _fft(float* data, int cols, int rows, bool forward);
	static CUFFT* _instance();
	bool _intialized;
};
