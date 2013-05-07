#include <CL/cl.h>
#include <boost/shared_array.hpp>

#include "surface_measurement_module.h"
#include "miroir_manager.h"
#include "gpu_manager.h"
#include "../src/OpenCL/gpu_def.h"

#include <pcl/io/openni_grabber.h>
#include <pcl/features/normal_3d.h>


// helper function TODO : put it somewhere
pcl::PointCloud<pcl::PointXYZ>::Ptr
convertToXYZPointCloud (pcl::OpenNIGrabber* grabber, const boost::shared_ptr<openni_wrapper::DepthImage>& depth_image)
{
	boost::shared_ptr< openni_wrapper::OpenNIDevice > device_ = grabber->getDevice();
	XnMapOutputMode depth_md = device_->getDepthOutputMode();
	std::string rgb_frame_id_ = "/openni_rgb_optical_frame";
	std::string depth_frame_id_ = "/openni_depth_optical_frame";
	unsigned depth_width_ = depth_md.nXRes;
	unsigned depth_height_ = depth_md.nYRes;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud <pcl::PointXYZ>);

	cloud->height = depth_height_;
	cloud->width = depth_width_;
	cloud->is_dense = false;

	cloud->points.resize (cloud->height * cloud->width);

	register float constant = 1.0f / grabber->getDevice()->getDepthFocalLength (depth_width_);

	if (device_->isDepthRegistered ())
		cloud->header.frame_id = rgb_frame_id_;
	else
		cloud->header.frame_id = depth_frame_id_;

	register int centerX = (cloud->width >> 1);
	int centerY = (cloud->height >> 1);

	float bad_point = std::numeric_limits<float>::quiet_NaN ();

	// we have to use Data, since operator[] uses assert -> Debug-mode very slow!
	register const unsigned short* depth_map = depth_image->getDepthMetaData ().Data ();
	if (depth_image->getWidth() != depth_width_ || depth_image->getHeight () != depth_height_)
	{
	static unsigned buffer_size = 0;
	static boost::shared_array<unsigned short> depth_buffer (0);

	if (buffer_size < depth_width_ * depth_height_)
	{
		buffer_size = depth_width_ * depth_height_;
		depth_buffer.reset (new unsigned short [buffer_size]);
	}
	depth_image->fillDepthImageRaw (depth_width_, depth_height_, depth_buffer.get ());
	depth_map = depth_buffer.get ();
	}

	register int depth_idx = 0;
	for (int v = -centerY; v < centerY; ++v)
	{
		for (register int u = -centerX; u < centerX; ++u, ++depth_idx)
		{
			pcl::PointXYZ& pt = cloud->points[depth_idx];
			// Check for invalid measurements
			if (depth_map[depth_idx] == 0 ||
				depth_map[depth_idx] == depth_image->getNoSampleValue () ||
				depth_map[depth_idx] == depth_image->getShadowValue ())
			{
				// not valid
				pt.x = pt.y = pt.z = bad_point;
				continue;
			}
			pt.z = depth_map[depth_idx] * 0.001f;
			pt.x = static_cast<float> (u) * pt.z * constant;
			pt.y = static_cast<float> (v) * pt.z * constant;
			
		}
	}
	cloud->sensor_origin_.setZero ();
	cloud->sensor_orientation_.w () = 0.0f;
	cloud->sensor_orientation_.x () = 1.0f;
	cloud->sensor_orientation_.y () = 0.0f;
	cloud->sensor_orientation_.z () = 0.0f;  
	return (cloud);
}

SurfaceMeasurementModule::SurfaceMeasurementModule( int w, int h, float focal_length )
{
	cloud_filtered = pcl::PointCloud< pcl::PointXYZRGBA >::Ptr( new pcl::PointCloud< pcl::PointXYZRGBA >() );
	cl_int err; 

	// gpu programs	
	measurement_vertices = GpuManager::createKernel("../src/OpenCL/bilateral.cl", "measurement_vertices");
	measurement_normals = GpuManager::createKernel("../src/OpenCL/bilateral.cl", "measurement_normals");

	dmap_h = new unsigned short[NPIX];
	vmap_h = new cl_float3[NPIX];
	nmap_h = new cl_float3[NPIX];

	float sig_s = 100.0f;
	float sig_r = 200.0f;
	/*int w = MiroirManager::getDepthWidth();
	int h = MiroirManager::getDepthHeight();
	float fl = MiroirManager::getFocalLength();*/

	sigma_s_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float), &sig_s, &err);
	assert(err == CL_SUCCESS);
	sigma_r_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float), &sig_r, &err);
	assert(err == CL_SUCCESS);

	vmap_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_WRITE, sizeof(cl_float3)*NPIX, NULL, &err);
	assert(err == CL_SUCCESS);
	nmap_d = clCreateBuffer(GpuManager::context, CL_MEM_WRITE_ONLY, sizeof(cl_float3)*NPIX, NULL, &err);
	assert(err == CL_SUCCESS);
}

//#define PROFILING

void SurfaceMeasurementModule::bilateralFilter( cl_mem raw_dmap )
{
	//const XnDepthPixel* depth_map = src.getDepthMetaData().Data();
	//int npixel = src.getWidth()*src.getHeight();
	//unsigned short* buffer = new unsigned short[npixel];
	cl_int err;
			
	//cl_mem res_d = clCreateBuffer(GpuManager::context, CL_MEM_WRITE_ONLY, sizeof(unsigned short)*NPIX, NULL, &err);
	//assert(err == CL_SUCCESS);
	
	cl_mem width_d = MiroirManager::getCLWidth();
	cl_mem height_d = MiroirManager::getCLHeight();
	cl_mem fl_d = MiroirManager::getCLFocalLength();

	cl_uint j = 0;
	err = clSetKernelArg(measurement_vertices, j++, sizeof(raw_dmap), (void*)&raw_dmap);
	err |= clSetKernelArg(measurement_vertices, j++, sizeof(width_d), (void*)&width_d);
	err |= clSetKernelArg(measurement_vertices, j++, sizeof(height_d), (void*)&height_d);
	err |= clSetKernelArg(measurement_vertices, j++, sizeof(fl_d), (void*)&fl_d);
	//err |= clSetKernelArg(measurement_vertices, j++, sizeof(res_d), (void*)&res_d);
	err |= clSetKernelArg(measurement_vertices, j++, sizeof(vmap_d), (void*)&vmap_d);
	err |= clSetKernelArg(measurement_vertices, j++, sizeof(sigma_s_d), (void*)&sigma_s_d);
	err |= clSetKernelArg(measurement_vertices, j++, sizeof(sigma_r_d), (void*)&sigma_r_d);

	assert(err == CL_SUCCESS);

	int width, height;
	MiroirManager::getSensorDimensions(width, height);

	size_t global_ws[2] = { width, height };

#ifdef PROFILING
	LARGE_INTEGER g_PerfFrequency; 
	LARGE_INTEGER g_PerformanceCountNDRangeStart; 
	LARGE_INTEGER g_PerformanceCountNDRangeStop; 
	QueryPerformanceFrequency(&g_PerfFrequency); 
	QueryPerformanceCounter(&g_PerformanceCountNDRangeStart);
#endif

	err = clEnqueueNDRangeKernel(GpuManager::queue, measurement_vertices, 2, 0, global_ws, NULL, 0, NULL, NULL);

	clFinish(GpuManager::queue);

#ifdef PROFILING
	QueryPerformanceCounter(&g_PerformanceCountNDRangeStop);  
	float seconds = (float)((g_PerformanceCountNDRangeStop.QuadPart - g_PerformanceCountNDRangeStart.QuadPart)/(float)g_PerfFrequency.QuadPart); 
#endif

	assert(err == CL_SUCCESS);
	// compute normals
	j = 0;
	err = clSetKernelArg(measurement_normals, j++, sizeof(width_d), (void*)&width_d);
	err |= clSetKernelArg(measurement_normals, j++, sizeof(height_d), (void*)&height_d);
	err |= clSetKernelArg(measurement_normals, j++, sizeof(vmap_d), (void*)&vmap_d);
	err |= clSetKernelArg(measurement_normals, j++, sizeof(nmap_d), (void*)&nmap_d);
	assert(err == CL_SUCCESS);

#ifdef PROFILING
	QueryPerformanceFrequency(&g_PerfFrequency); 
	QueryPerformanceCounter(&g_PerformanceCountNDRangeStart);
#endif

	err = clEnqueueNDRangeKernel(GpuManager::queue, measurement_normals, 2, 0, global_ws, NULL, 0, NULL, NULL);

	clFinish(GpuManager::queue);

#ifdef PROFILING
	QueryPerformanceCounter(&g_PerformanceCountNDRangeStop);  
	seconds = (float)((g_PerformanceCountNDRangeStop.QuadPart - g_PerformanceCountNDRangeStart.QuadPart)/(float)g_PerfFrequency.QuadPart); 
#endif

	assert(err == CL_SUCCESS);
}

void SurfaceMeasurementModule::processData()
{
	/*	1 - bilateral filtering of the raw depth map
		2 - back project the filtered depth map to sensor's frame of reference to obtain thwe VertexMap -> convertToXYZPointCloud
		3 - compute NormalMap from neighbouring vertices

		Note : use vertex validity mask Mk(u) = 1 if depth measurement is valid, Mk(u) = 0, if 
		depth measurement if missing.

		4 - compute L = 3 "mimaps" of VertexMap and NormalMap by block averaging and subsampling to half
		the resolution of previous level.
	*/ 

	bilateralFilter( dmap );
	
	// create the cloud from the vertex map
	//cloud_filtered = convertToXYZPointCloud( static_cast<pcl::OpenNIGrabber*>(MiroirManager::getGrabber()), dm_filtered );
#ifndef MIROIR_GL_INTEROP
	if(MiroirManager::getShowMode() == DEPTH_MAP_BIL)
	{
		boost::mutex::scoped_lock lock(MiroirManager::getMutex());
		clEnqueueReadBuffer(GpuManager::queue, vmap_d, CL_TRUE, 0, sizeof(cl_float3)*NPIX, vmap_h, 0, NULL, NULL);
		clEnqueueReadBuffer(GpuManager::queue, nmap_d, CL_TRUE, 0, sizeof(cl_float3)*NPIX, nmap_h, 0, NULL, NULL);
		pcl::OpenNIGrabber* grabber = static_cast<pcl::OpenNIGrabber*>(MiroirManager::getGrabber());
		cloud_filtered = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud <pcl::PointXYZRGBA>);
		cloud_filtered->height = grabber->getDevice()->getDepthOutputMode().nYRes;
		cloud_filtered->width = grabber->getDevice()->getDepthOutputMode().nXRes;
		cloud_filtered->is_dense = false;

		cloud_filtered->points.resize (cloud_filtered->height * cloud_filtered->width);

		if (grabber->getDevice()->isDepthRegistered ())
			cloud_filtered->header.frame_id = "/openni_rgb_optical_frame";
		else
			cloud_filtered->header.frame_id = "/openni_depth_optical_frame";

		for(int i=0; i<NPIX; i++)
		{
			pcl::PointXYZRGBA& pt = cloud_filtered->points[i];

			if(vmap_h[i].s[2] == 0)
			{
				pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN ();
				continue;
			}
		
			pt.x = vmap_h[i].s[0] * 0.001;
			pt.y = vmap_h[i].s[1] * 0.001;
			pt.z = vmap_h[i].s[2] * 0.001;
			pt.r = 255 * (nmap_h[i].s[0] + 1) / 2;
			pt.g = 255 * (nmap_h[i].s[1] + 1) / 2;
			pt.b = 255 * (nmap_h[i].s[2] + 1) / 2;
		}

		cloud_filtered->sensor_origin_.setZero ();
		/*cloud_filtered->sensor_orientation_.w () = 1.0f;
		cloud_filtered->sensor_orientation_.x () = 0.0f;
		cloud_filtered->sensor_orientation_.y () = 0.0f;
		cloud_filtered->sensor_orientation_.z () = 0.0f;*/  
	}
#endif
}
