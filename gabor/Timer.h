#pragma once

#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

class Timer
{
public:
	inline void start()
	{
		gettimeofday(&_startTime, NULL);
	}
	inline double stop()
	{
		gettimeofday(&_endTime, NULL);
		_dt = (_endTime.tv_sec * 1000000 + _endTime.tv_usec) -
		      (_startTime.tv_sec * 1000000 + _startTime.tv_usec);
		return _dt / 1000000.0;
	}
	inline unsigned long days() const { return _dt / 86400000000; }
	inline unsigned long hours() const { return _dt / 3600000000; }
	inline unsigned long minutes() const { return _dt / 60000000; }
	inline unsigned long seconds() const { return _dt / 1000000; }
	inline unsigned long milliseconds() const { return _dt / 1000; }
	inline unsigned long microseconds() const { return _dt; }
protected:
	struct timeval _startTime;
	struct timeval _endTime;
	unsigned long _dt;
};
