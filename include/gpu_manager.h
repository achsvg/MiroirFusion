#ifndef GPU_MANAGER_H
#define GPU_MANAGER_H

#include <CL/cl.h>
#include <gl/glew.h>
#include "../src/OpenCL/gpu_def.h"
#include <boost/thread/thread.hpp>
#if defined (__APPLE__) || defined(MACOSX)
	static const char* CL_GL_SHARING_EXT = "cl_APPLE_gl_sharing";
#else
	static const char* CL_GL_SHARING_EXT = "cl_khr_gl_sharing";
#endif

/** \brief Manages the gpu programs
  * \author Anthony Chansavang <anthony.chansavang@gmail.com>
  */
class GpuManager
{
public:
	
	static cl_context context;
	static cl_command_queue queue;
	static cl_device_id device;
	static cl_int error;
	static cl_platform_id platforms;

	static cl_ulong max_alloc_size;

	static void initCL(bool kernel_recompile);
	static void cleanCL();
	static char* loadProgSource(const char* filename);
	
	static GLuint createProgram(char* vert_filename, char* frag_filename, GLuint& vshader, GLuint& fshader);
	static cl_kernel createKernel(const char* filename, const char* kernel_name, bool fastMath=false);
	static cl_mem createSharedBuffer(GLsizeiptr size, const void* data, cl_mem_flags flags);
	static void createSharedTexture(int width, int height, cl_mem_flags flags, cl_mem& cl_tex, GLuint& gl_tex);

	static void __stdcall contextNotify(const char *errinfo, const void *private_info, size_t cb, void *user_data);
	static void __stdcall buildProgramNotify(cl_program program, void* user_data);

	static boost::mutex& getMutex() { return mutex; }
private:
	static bool cl_init;
	static bool gl_init;

	static boost::mutex mutex;
};

#endif