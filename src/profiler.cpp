#include <boost/timer/timer.hpp>
#include "profiler.h"

//MiroirProfiler& MiroirProfiler::profiler = MiroirProfiler();

using boost::timer::cpu_timer;

std::map<std::string, cpu_timer*> MiroirProfiler::start_time = std::map<std::string, cpu_timer*>();
bool MiroirProfiler::enabled = false;

MiroirProfiler::MiroirProfiler()
{
}

//MiroirProfiler& MiroirProfiler::getSingleton()
//{
//	return profiler;
//}

void MiroirProfiler::start(const std::string& name)
{
	if(enabled)
	{
		cpu_timer* timer = new cpu_timer();
		timer->start();
		start_time[name] = timer;
	}
}

void MiroirProfiler::stop(const std::string& name)
{
	if(enabled)
	{
		auto pos = start_time.find(name);
		if( pos == start_time.end())
			return;
	
		cpu_timer* timer = start_time[name];
		start_time[name]->stop();
		std::cout << name << " elapsed time : " << boost::timer::format(timer->elapsed(), 5, "%t") << std::endl;

		delete timer;
		start_time.erase(pos);
	}
}
