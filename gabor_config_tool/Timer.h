
/*
* File name: Timer.h
*/

#ifndef __TIMER_H__
#define __TIMER_H__

#include <sys/time.h>

using namespace std;

/**
* Basic timer class. Uses gettimeofday() function.
*/
class Timer
{
public:
	/**
	* Defalaut constructor.
	*/
	Timer() {}

	/**
	* Default destructor.
	*/
	~Timer(){}

	/**
	* "Start" timer: save current time.
	*/
	void start()
	{
		gettimeofday( &_start, 0 );
	}

	/**
	* "Stop" timer: get current time and subtract from start time.
	* Returns microseconds since start().
	*/
	unsigned long stop()
	{
		struct timeval now;
		gettimeofday( &now, 0 );
		return (now.tv_sec-_start.tv_sec)*1000000+(now.tv_usec-_start.tv_usec);
	}

private:

	struct timeval _start;

};

#endif

