#include "gpu_manager.h"

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <CL/cl_gl.h>
#include <stdlib.h>

#include <boost/filesystem.hpp>

#if defined WIN32 || defined _WIN32 || defined _WIN64
#	include <windows.h>
#	include <Shlobj.h>
//#	include <WinDef.h>
//#	include <Wingdi.h>
#elif defined __UNIX__
#	include <GL/glx.h>
#elif defined __APPLE__
// include wut here?
#endif

typedef CL_API_ENTRY cl_int
(CL_API_CALL *clGetGLContextInfoKHR_fn)(const cl_context_properties *properties,
                                        cl_gl_context_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret);

#define clGetGLContextInfoKHR clGetGLContextInfoKHR_proc
static clGetGLContextInfoKHR_fn clGetGLContextInfoKHR;

cl_context GpuManager::context;
cl_command_queue GpuManager::queue;
cl_device_id GpuManager::device;
cl_int GpuManager::error;
cl_platform_id GpuManager::platforms;
bool GpuManager::cl_init = false;
bool GpuManager::gl_init = false;
boost::mutex GpuManager::mutex;
cl_ulong GpuManager::max_alloc_size;

cl_mem GpuManager::createSharedBuffer(GLsizeiptr size, const void* data, cl_mem_flags flags)
{
	GLuint gl_buffer_id;
	 // Create a buffer object in OpenGL and allocate space
    glGenBuffers(1, &gl_buffer_id);
	glBindBuffer(GL_ARRAY_BUFFER, gl_buffer_id);
	// Note: specify GL_STATIC_DRAW_ARB to modify outside of GL
	glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW); 

	// Note: could use colors, normals, etc
	//glVertexPointer(4, GL_FLOAT, 0, 0);		// render the buffer
	//glBindBuffer( GL_ARRAY_BUFFER_ARB, 0);	// is that necessary?

	// Create a reference cl_mem object from GL buffer object
	int err;
	cl_mem buf = clCreateFromGLBuffer(context, flags, gl_buffer_id, &err);
	assert(err == CL_SUCCESS);
	return buf;
}

void GpuManager::createSharedTexture(int width, int height, cl_mem_flags flags, cl_mem& cl_tex, GLuint& gl_tex)
{
	boost::mutex::scoped_lock tex_lock(GpuManager::getMutex());
	glGetError();
	//GLuint tex_screen;
	GLenum gl_err;

	// Create GL texture
	glGenTextures(1, &gl_tex); 
	gl_err = glGetError();
	assert(gl_err != GL_INVALID_VALUE && gl_err != GL_INVALID_OPERATION);

	glBindTexture(GL_TEXTURE_2D, gl_tex);

	// Set texture parameters
	// Poor filtering. Needed !
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	gl_err = glGetError();
	assert(gl_err == GL_NO_ERROR);

	// Setup data storage
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	gl_err = glGetError();
	assert(gl_err == GL_NO_ERROR);
	//glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex_screen, 0);

	//assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE);

	// Create CL image from Screen Texture – the CL kernel will write to this
	cl_int err;
	//cl_mem cl_screen;
	cl_tex = clCreateFromGLTexture(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, gl_tex, &err );
	assert(err == CL_SUCCESS);

	//err = clEnqueueAcquireGLObjects(GpuManager::queue, 1, &cl_tex, 0,0,0);
	//assert(err == CL_SUCCESS);
	//return cl_screen;
	//GLuint fbo;
	//GLuint rb_color;
	//int err;

	//// Create and bind the FBO
	//glGenFramebuffers(1, &fbo);
	//glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	//// Create a RGBA8 render buffer
	//glGenRenderbuffers(1, &rb_color);
	//glBindRenderbuffer(GL_RENDERBUFFER, rb_color);
	//glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height);
	//// Attach it as color attachment to the FBO
	//glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb_color);

	//// Create the CL image from the color renderbuffer – will read from this in the kernel
	//cl_mem cl_scene; 
	//cl_scene = clCreateFromGLRenderbuffer(context, flags, rb_color, &err);
	//assert(err == CL_SUCCESS);
	//// CL can query properties on this image as with normal CL images
	//cl_image_format image_format;
	//clGetImageInfo (cl_scene, CL_IMAGE_FORMAT, sizeof(cl_image_format), &image_format, NULL);
	//// image_format will be CL_UNSIGNED_INT8, CL_BGRA

	//return cl_scene;
}

char* GpuManager::loadProgSource( char* filename )
{
	FILE* fh;
	char* source;
	//int fd;

	fh = fopen(filename, "rb");
	if (fh == 0)
		return 0;

	//fseek(fh, 0, SEEK_END);
	//int size = ftell(fh);
	//fseek(fh, 0, SEEK_SET);

	struct _stat filestatus;

	if( _stat (filename, &filestatus) == -1 )
		return 0;

	//source = new char[filestatus.st_size + 1];
	int size = filestatus.st_size;

	source = new char[size + 1];

	fread(source, size, 1, fh);
	source[size] = '\0';

	return source;
}

void GpuManager::initCL(bool kernel_recompile)
{
	if(!cl_init)
	{
		if(kernel_recompile)
		{
			#ifdef WIN32
				char appdata[MAX_PATH];
				SHGetFolderPath( NULL, 
								 CSIDL_APPDATA, 
								 NULL, 
								 0, 
								 appdata);
				std::string cc = appdata;
				cc += "\\NVIDIA\\ComputeCache";
				// clear compute cache folder if NVIDIA card
				if(boost::filesystem::exists(cc))
				{
					boost::filesystem::remove_all(cc);
					boost::filesystem::create_directory(cc);
				}
				else
					SetEnvironmentVariable("CUDA_CACHE_DISABLE", "1");
			#else
				SetEnvironmentVariable("CUDA_CACHE_DISABLE", "1");
			#endif
		}
#ifdef MIROIR_GL_INTEROP
		error = clGetPlatformIDs( 1, &platforms, NULL );
		assert(error == CL_SUCCESS);
		clGetDeviceIDs(platforms, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);

		assert(device != NULL);

		// Get string containing supported device extensions
		size_t ext_size = 0;
		char* ext_string = new char[1024];
		error = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 1024, ext_string, NULL);
		//clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_size, ext_string, NULL);
		std::string ext = ext_string;

		if(ext.find(CL_GL_SHARING_EXT) == std::string::npos)
		{
			//extension not found!
			std::cout << "CL/GL sharing extension not found! " << ext_string << std::endl;
			delete [] ext_string;
			return;
		}
		delete [] ext_string;

		// Create CL context properties, add WGL context & handle to DC 
		cl_context_properties properties[] = { 
	#if defined WIN32 || defined _WIN32 || defined _WIN64
			CL_GL_CONTEXT_KHR,   (cl_context_properties)wglGetCurrentContext(), // WGL Context 
			CL_WGL_HDC_KHR,      (cl_context_properties)wglGetCurrentDC(),      // WGL HDC
	#elif defined __unix__
			CL_GL_CONTEXT_KHR,   (cl_context_properties)glXGetCurrentContext(), // GLX Context 
			CL_GLX_DISPLAY_KHR,  (cl_context_properties)glXGetCurrentDisplay(), // GLX Display
	#endif
			CL_CONTEXT_PLATFORM, (cl_context_properties)platforms,               // OpenCL platform
			0
		};

		// get ptr to extension func
		clGetGLContextInfoKHR = (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddress("clGetGLContextInfoKHR");
        if (!clGetGLContextInfoKHR)
        {
            std::cerr << "Error getting clGetGLContextInfoKHR function ptr.\n";
            return;
        }

		// Find CL capable devices in the current GL context 
		cl_device_id devices[32]; size_t size;
		error = clGetGLContextInfoKHR(properties, CL_DEVICES_FOR_GL_CONTEXT_KHR, 32 * sizeof(cl_device_id), devices, &size);
		assert(error == CL_SUCCESS);

		clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

		// Create a context using the supported devices
		int count = size / sizeof(cl_device_id);
		context = clCreateContext(properties, count, devices, NULL, 0, &error);
		assert(error == CL_SUCCESS);

		clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL);
		queue = clCreateCommandQueue(context, device, 0, &error);
		assert(error == CL_SUCCESS);
#else
////////////////////////// INIT WITHOUT OPENGL //////////////////////////////////////////////
		error = clGetPlatformIDs( 1, &platforms, NULL );
		assert(error == CL_SUCCESS);

		clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

		//context = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &error); 
		context = clCreateContext(0, 1, &device, GpuManager::contextNotify, NULL, &error);
		assert(error == CL_SUCCESS);

		clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL);
		queue = clCreateCommandQueue(context, device, 0, &error);
		assert(error == CL_SUCCESS);
//////////////////////////////////////////////////////////////////////////////////////////////
#endif
		clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_alloc_size, NULL);
		std::cout << "maximum allocation size: " << max_alloc_size << std::endl;

		cl_ulong local_mem;
		clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem, NULL); 
		std::cout << "local memory size: " << local_mem << std::endl;

		cl_ulong max_wrk_grp;
		clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_ulong), &max_wrk_grp, NULL);
		std::cout << "max work group size (max threads per block): " << max_wrk_grp << std::endl;

		cl_init = true;
	}
}

void GpuManager::cleanCL()
{
	clReleaseCommandQueue(GpuManager::queue);
	clReleaseContext(GpuManager::context);
}

void __stdcall GpuManager::contextNotify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
	std::cout << errinfo << std::endl;
}

void __stdcall GpuManager::buildProgramNotify(cl_program program, void* user_data)
{
	std::cout << "Build" << std::endl;
}

GLuint GpuManager::createProgram(char* vert_filename, char* frag_filename, GLuint& vshader, GLuint& fshader)
{
	glGetError();
	GLuint program;
	vshader = glCreateShader(GL_VERTEX_SHADER);
	fshader = glCreateShader(GL_FRAGMENT_SHADER);
	GLenum err = glGetError();
	assert(err == GL_NO_ERROR);
	
	const char* vsource[] = {
		loadProgSource(vert_filename)
	};

	const char* fsource[] = {
		loadProgSource(frag_filename)
	};

	glShaderSource(vshader, 1, (const GLchar**)&vsource, NULL);
	err = glGetError();
	assert(err == GL_NO_ERROR);
	glShaderSource(fshader, 1, (const GLchar**)&fsource, NULL);
	err = glGetError();
	assert(err == GL_NO_ERROR);
	glCompileShader(vshader);
	err = glGetError();
	assert(err == GL_NO_ERROR);
	glCompileShader(fshader);
	err = glGetError();
	assert(err == GL_NO_ERROR);

	GLint vstatus, fstatus;

	glGetProgramiv(vshader, GL_COMPILE_STATUS, &vstatus);
	glGetProgramiv(fshader, GL_COMPILE_STATUS, &fstatus);
	if (!vstatus || !fstatus)
	{
	    GLint blen = 0;	
		GLsizei slen = 0;
		GLchar* compiler_log;

		glGetShaderiv(vshader, GL_INFO_LOG_LENGTH , &blen);       
		
		compiler_log = new GLchar[blen];
		glGetInfoLogARB(vshader, blen, &slen, compiler_log);
		std::cout << "vert shader compiler log:\n" << compiler_log;
		delete [] compiler_log;

		glGetShaderiv(fshader, GL_INFO_LOG_LENGTH , &blen);       
		
		compiler_log = new GLchar[blen];
		glGetInfoLogARB(fshader, blen, &slen, compiler_log);
		std::cout << "frag shader compiler log:\n" << compiler_log;
		delete [] compiler_log;

		return -1;
	}    
	glGetError();

	program = glCreateProgram();
	glAttachShader(program, vshader);
	glAttachShader(program, fshader);
	glLinkProgram(program);
	err = glGetError();
	assert(err == GL_NO_ERROR);

	glGetProgramiv(vshader, GL_COMPILE_STATUS, &vstatus);
	glGetProgramiv(fshader, GL_COMPILE_STATUS, &fstatus);
	if (!vstatus || !fstatus)
	{
	    GLint blen = 0;	
		GLsizei slen = 0;
		GLchar* compiler_log;

		glGetShaderiv(vshader, GL_INFO_LOG_LENGTH , &blen);       
		
		compiler_log = new GLchar[blen];
		glGetInfoLogARB(vshader, blen, &slen, compiler_log);
		std::cout << "vert shader compiler log:\n" << compiler_log;
		delete [] compiler_log;

		glGetShaderiv(fshader, GL_INFO_LOG_LENGTH , &blen);       
		
		compiler_log = new GLchar[blen];
		glGetInfoLogARB(fshader, blen, &slen, compiler_log);
		std::cout << "frag shader compiler log:\n" << compiler_log;
		delete [] compiler_log;

		return -1;
	}    

	return program;
}

cl_kernel GpuManager::createKernel(char* filename, const char* kernel_name, bool fastMath)
{
	cl_int err; 

	const char* source[] = 
	{
		GpuManager::loadProgSource( filename )
	};

	cl_program program = clCreateProgramWithSource(GpuManager::context, 1, source, NULL, &err);
	assert(err == CL_SUCCESS);
		
	// memory usage report: -cl-nv-verbose
	std::string arg = /*"-cl-mad-enable"*/"-I ../src/OpenCL -Werror";
	if(fastMath) arg += " -cl-mad-enable -cl-fast-relaxed-math";
	err = clBuildProgram(program, 1, &(GpuManager::device), arg.c_str(), NULL, NULL);
	
	// Shows the log
	char* build_log;
	size_t log_size;
	// First call to know the proper size
	clGetProgramBuildInfo(program, GpuManager::device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	build_log = new char[log_size+1];
	// Second call to get the log
	clGetProgramBuildInfo(program, GpuManager::device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
	build_log[log_size] = '\0';
	std::cout << build_log << std::endl;
	delete[] build_log;

	assert(err == CL_SUCCESS);

	cl_build_status build_status;
	clGetProgramBuildInfo(program, GpuManager::device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
	assert(build_status == CL_BUILD_SUCCESS);

	// extract the kernel
	cl_kernel k = clCreateKernel(program, kernel_name, &err);
	assert(err == CL_SUCCESS);

	//err = clReleaseProgram(program);
	//assert(err == CL_SUCCESS);

	return k;
}