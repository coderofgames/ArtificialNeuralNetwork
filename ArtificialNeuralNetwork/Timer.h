#ifndef TIMER_H
#define TIMER_H

#include "Windows.h"
#include "mmsystem.h"

class Timer
{
public:
	Timer(){
		bActive = false;
		Reset();
	}
	~Timer(){}

	void Start()
	{
		bActive = true;
		Update();

	}
	void Stop()
	{
		bActive = false;
	}


	void Reset(){
		timeDelta = 0.0;
		last_time = 0.0;
		current_time = 0.0;
		total_time = 0.0;
	}

	double Update()
	{
		if (!bActive)
			return 0.0;


#ifndef WIN32
#define WIN32
#endif
#ifdef WIN32
		static __int64 gTime, gLastTime;
		__int64 freq;
		QueryPerformanceCounter((LARGE_INTEGER *)&gTime);  // Get current count
		QueryPerformanceFrequency((LARGE_INTEGER *)&freq); // Get processor freq
		timeDelta = (double)(gTime - gLastTime) / (double)freq;
		gLastTime = gTime;
		last_time = (double)gTime;
		current_time = (double)gLastTime;
		last_time = current_time;
		total_time += timeDelta;
#else
		struct timeval tv;
		static struct timeval lasttv = { 0, 0 };
		if (lasttv.tv_usec == 0 && lasttv.tv_sec == 0)
			gettimeofday(&lasttv, NULL);
		gettimeofday(&tv, NULL);
		timeDelta = (tv.tv_usec - lasttv.tv_usec) / 1000000.f
			+ (tv.tv_sec - lasttv.tv_sec);
		lasttv = tv;
#endif
		return timeDelta;
	}

	double GetTimeInSeconds(){
		return current_time;
	}
	double GetTimeDelta(){
		return timeDelta;
	}



	bool bActive;
	double timeDelta;
	double last_time;
	double current_time;
	double total_time;
};
#endif