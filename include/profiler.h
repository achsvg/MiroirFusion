#ifndef MIROIR_PROFILER
#define MIROIR_PROFILER

#include <map>

namespace boost
{
	namespace timer{ class cpu_timer; }
}
class MiroirProfiler
{
public:
	//MiroirProfiler& getSingleton();
	static void enable(){enabled = true;}
	static void disable(){enabled = false;}
	static void start( const std::string& name );
	static void stop( const std::string& name );
private:
	//static MiroirProfiler& profiler;
	static std::map<std::string, boost::timer::cpu_timer*> start_time;
	static bool enabled;

	MiroirProfiler();
};

#endif
