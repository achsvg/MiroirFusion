#include "raycast_module.h"
#include "gpu_manager.h"
#include "miroir_manager.h"
#include "../src/OpenCL/gpu_def.h"
#include <pcl/io/openni_grabber.h>
#include <CL/cl_gl.h>

RaycastModule::RaycastModule()
{	
	raycast_kernel = GpuManager::createKernel((std::string(KERNEL_PATH)+std::string("raycast.cl")).c_str(), "raycast");

	int width, height;
	MiroirManager::getSensorDimensions(width, height);

	int npixel = width*height;
	int err;

	vmap_h = new cl_float3[npixel];
	nmap_h = new cl_float3[npixel];

	vmap_d = clCreateBuffer(GpuManager::context, CL_MEM_WRITE_ONLY, sizeof(cl_float3)*npixel, NULL, &err);
	nmap_d = clCreateBuffer(GpuManager::context, CL_MEM_WRITE_ONLY, sizeof(cl_float3)*npixel, NULL, &err);
	assert(err == CL_SUCCESS);

#ifdef MIROIR_GL_INTEROP
	GpuManager::createSharedTexture(width, height, CL_MEM_WRITE_ONLY, tex_d, gl_tex);
#endif

	tex_dim[0] = width;
	tex_dim[1] = height;
}

RaycastModule::~RaycastModule()
{
	clReleaseMemObject(vmap_d);
	clReleaseMemObject(nmap_d);
	clReleaseKernel(raycast_kernel);
	delete [] vmap_h;
	delete [] nmap_h;
}

//#define PROFILING

void RaycastModule::raycast(cl_mem TSDF, cl_float4 cam_pos[4])
{
	boost::mutex::scoped_lock lock(raycast_mutex);
	cl_int err;

	int width, height;
	MiroirManager::getWindowDimensions(width, height);		
	int npixel = width*height;

	if(width != tex_dim[0] || height != tex_dim[1])
		return;

	cl_float2 princ_pt = MiroirManager::getPrincipalPoint();

	cl_mem tsdf_params = MiroirManager::getCLTSDFParams();

	cl_mem cam_pos_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float4)*4, cam_pos, &err);
	assert(err == CL_SUCCESS);
	cl_mem width_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &width,&err);
	assert(err == CL_SUCCESS);
	cl_mem height_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &height,&err);
	assert(err == CL_SUCCESS);
	cl_mem fl_d = MiroirManager::getCLFocalLength();//clCreateBuffer(GpuManager::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float), &fl, &err);
	//assert(err == CL_SUCCESS);
	cl_mem ppt_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float2), &princ_pt, &err);
	assert(err == CL_SUCCESS);
	
	cl_uint j = 0;

	// GL-CL sync
	glFlush();
	glFinish();

	// All pending GL calls have finished -> safe to acquire the buffer in CL
	//clEnqueueAcquireGLObjects(GpuManager::queue, 1, &vmap_d, 0,0,0);
	//clEnqueueAcquireGLObjects(GpuManager::queue, 1, &nmap_d, 0,0,0);

	err |= clSetKernelArg(raycast_kernel, j++, sizeof(cl_mem), (void*)&TSDF);
	err |= clSetKernelArg(raycast_kernel, j++, sizeof(cl_mem), (void*)&width_d);
	err |= clSetKernelArg(raycast_kernel, j++, sizeof(cl_mem), (void*)&height_d);
	err |= clSetKernelArg(raycast_kernel, j++, sizeof(cl_mem), (void*)&cam_pos_d);
	err |= clSetKernelArg(raycast_kernel, j++, sizeof(cl_mem), (void*)&fl_d);
	err |= clSetKernelArg(raycast_kernel, j++, sizeof(cl_mem), (void*)&ppt_d);
	err |= clSetKernelArg(raycast_kernel, j++, sizeof(cl_mem), (void*)&vmap_d);
	err |= clSetKernelArg(raycast_kernel, j++, sizeof(cl_mem), (void*)&nmap_d);
#ifdef MIROIR_GL_INTEROP
	
	err = clEnqueueAcquireGLObjects(GpuManager::queue, 1, &tex_d, 0,0,0);
	assert(err == CL_SUCCESS);
	err |= clSetKernelArg(raycast_kernel, j++, sizeof(cl_mem), &tex_d);
	
#endif
	err |= clSetKernelArg(raycast_kernel, j++, sizeof(cl_mem), (void*)&tsdf_params);

	assert(err == CL_SUCCESS);

	size_t global_ws[2] = { width, height };

#ifdef PROFILING
	LARGE_INTEGER g_PerfFrequency; 
	LARGE_INTEGER g_PerformanceCountNDRangeStart; 
	LARGE_INTEGER g_PerformanceCountNDRangeStop; 
	QueryPerformanceFrequency(&g_PerfFrequency); 
	QueryPerformanceCounter(&g_PerformanceCountNDRangeStart);
#endif

	err = clEnqueueNDRangeKernel(GpuManager::queue, raycast_kernel, 2, 0, global_ws, NULL, 0, NULL, NULL);
	clFinish(GpuManager::queue);

#ifdef PROFILING
	QueryPerformanceCounter(&g_PerformanceCountNDRangeStop);  
	float seconds = (float)((g_PerformanceCountNDRangeStop.QuadPart - g_PerformanceCountNDRangeStart.QuadPart)/(float)g_PerfFrequency.QuadPart); 
#endif

 	assert(err == CL_SUCCESS);

	//
	//cl_float3* test = new cl_float3[307200];
	//err = clEnqueueReadBuffer(GpuManager::queue, vmap_d, CL_TRUE, 0, sizeof(cl_float3)*npixel, vmap_h, 0, NULL, NULL);
	//err = clEnqueueReadBuffer(GpuManager::queue, nmap_d, CL_TRUE, 0, sizeof(cl_float3)*npixel, nmap_h, 0, NULL, NULL);
#ifdef MIROIR_GL_INTEROP
	err = clEnqueueReleaseGLObjects(GpuManager::queue, 1, &tex_d, 0,0,0);
	//err = clEnqueueReleaseGLObjects(GpuManager::queue, 1, &vmap_d, 0,0,0);
	//err |= clEnqueueReleaseGLObjects(GpuManager::queue, 1, &nmap_d, 0,0,0);
	assert(err == CL_SUCCESS);
#endif
	clFinish(GpuManager::queue);

	clReleaseMemObject(cam_pos_d);
	//clReleaseMemObject(width_d);
	//clReleaseMemObject(fl_d);
	clReleaseMemObject(ppt_d);
	//clReleaseMemObject(vmap_d);
	//clReleaseMemObject(nmap_d);
}

void RaycastModule::processData()
{
	raycast( tsdf, cam_pose );
	int width, height;
	MiroirManager::getSensorDimensions(width, height);
	int npixel = width*height;
	
	if(MiroirManager::getShowMode() == RAYCAST)
	{
		boost::mutex::scoped_lock lock(MiroirManager::getMutex());
		clEnqueueReadBuffer(GpuManager::queue, vmap_d, CL_TRUE, 0, sizeof(cl_float3)*npixel, vmap_h, 0, NULL, NULL);
		clEnqueueReadBuffer(GpuManager::queue, nmap_d, CL_TRUE, 0, sizeof(cl_float3)*npixel, nmap_h, 0, NULL, NULL);
		clFinish(GpuManager::queue);

		pcl::OpenNIGrabber* grabber = static_cast<pcl::OpenNIGrabber*>(MiroirManager::getGrabber());
		cloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud <pcl::PointXYZRGBA>);
		cloud->height = grabber->getDevice()->getDepthOutputMode().nYRes;
		cloud->width = grabber->getDevice()->getDepthOutputMode().nXRes;
		cloud->is_dense = false;

		cloud->points.resize (cloud->height * cloud->width);

		if (grabber->getDevice()->isDepthRegistered ())
			cloud->header.frame_id = "/openni_rgb_optical_frame";
		else
			cloud->header.frame_id = "/openni_depth_optical_frame";

		for(int i=0; i<cloud->height * cloud->width; i++)
		{
			pcl::PointXYZRGBA& pt = cloud->points[i];

			if(vmap_h[i].s[0] != vmap_h[i].s[0])
			{
				pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN ();
				continue;
			}
			
			pt.z = vmap_h[i].s[2] * 0.001;
			pt.x = vmap_h[i].s[0] * 0.001;
			pt.y = vmap_h[i].s[1] * 0.001;
			pt.r = pt.g = pt.b = 127;
			pt.r = 255 * (nmap_h[i].s[0] + 1) / 2;
			pt.g = 255 * (nmap_h[i].s[1] + 1) / 2;
			pt.b = 255 * (nmap_h[i].s[2] + 1) / 2;

		}

		cloud->sensor_origin_.setZero ();
		cloud->sensor_orientation_.w () = 0.0f;
		cloud->sensor_orientation_.x () = 1.0f;
		cloud->sensor_orientation_.y () = 0.0f;
		cloud->sensor_orientation_.z () = 0.0f;
	}
}

void RaycastModule::onWindowResized(int width, int height)
{
	boost::mutex::scoped_lock lock(raycast_mutex);
	//ost::mutex::scoped_lock lock(MiroirManager::getMutex());

	tex_dim[0] = width;
	tex_dim[1] = height;

#ifdef MIROIR_GL_INTEROP
	//glDeleteTextures(1, &gl_tex);

	//// creating the texture to render full screen quad
	//glActiveTexture(GL_TEXTURE0);
	//GpuManager::createSharedTexture(width, height, CL_MEM_WRITE_ONLY, tex_d, gl_tex);

	//int npixel = width*height;

	//clReleaseMemObject(vmap_d);
	//clReleaseMemObject(nmap_d);

	//int err;
	//vmap_d = clCreateBuffer(GpuManager::context, CL_MEM_WRITE_ONLY, sizeof(cl_float3)*npixel, NULL, &err);
	//nmap_d = clCreateBuffer(GpuManager::context, CL_MEM_WRITE_ONLY, sizeof(cl_float3)*npixel, NULL, &err);

	//delete [] vmap_h;
	//delete [] nmap_h;

	//vmap_h = new cl_float3[npixel];
	//nmap_h = new cl_float3[npixel];
#endif
}