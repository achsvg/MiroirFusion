#include "definitions.h"
#include "util.h"
#include "update_reconstruction_module.h"
#include "gpu_manager.h"
#include "miroir_manager.h"

#include <pcl/io/openni_camera/openni_depth_image.h>
#include <pcl/io/openni_grabber.h>
#include <limits>

UpdateReconstructionModule::UpdateReconstructionModule() : current_frame(0), enable(true)
{
	kernel = GpuManager::createKernel("../src/OpenCL/TSDF.cl", "TSDF");

	// TSDF stored in GPU
	cl_int err;

	cl_short2* buf = new cl_short2[(int)(RESX*RESY*RESZ)];

	for(int i=0; i<RESX*RESY*RESZ; i++)
	{
		buf[i].s[0] = SHORT_NAN; //std::numeric_limits<short>::quiet_NaN(); 
		buf[i].s[1] = -SHORT_MAX;
	}

	TSDF_d = clCreateBuffer(GpuManager::context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(cl_short2)*RESX*RESY*RESZ, buf, &err);
	assert(err == CL_SUCCESS);
	clEnqueueWriteBuffer(GpuManager::queue, TSDF_d, true, 0, sizeof(cl_short2)*RESX*RESY*RESZ, buf, 0, NULL, NULL);
	
	delete [] buf;

	cam_pose = new cl_float4[4];
}

UpdateReconstructionModule::~UpdateReconstructionModule()
{
	clReleaseMemObject(TSDF_d);
	clReleaseKernel(kernel);
}

//#define PROFILING

void UpdateReconstructionModule::TSDF()
{
	cl_int err;

	cl_float4 cam_pos_inv_h[4];// = {*(cam_pos_inv[0]), *(cam_pos_inv[1]), *(cam_pos_inv[2]), *(cam_pos_inv[3])}; 

	// transpose rotation matrix to get inverse
	MiroirUtil::invertCamTransform(cam_pose, cam_pos_inv_h);

	//cl_mem cam_pos_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float4)*4, cam_pose, &err);
	//assert(err == CL_SUCCESS);
	cl_mem cam_pos_inv_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float4)*4, cam_pos_inv_h, &err);
	assert(err == CL_SUCCESS);
	cl_mem k_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_ulong), &current_frame, &err);
	assert(err == CL_SUCCESS);

	cl_mem width_d = MiroirManager::getCLWidth();
	cl_mem height_d = MiroirManager::getCLHeight();
	cl_mem fl_d = MiroirManager::getCLFocalLength();
	cl_mem tsdf_params = MiroirManager::getCLTSDFParams();

	cl_uint j = 0;
	err = clSetKernelArg(kernel, j++, sizeof(TSDF_d), (void*)&TSDF_d);
	err |= clSetKernelArg(kernel, j++, sizeof(width_d), (void*)&width_d);
	err |= clSetKernelArg(kernel, j++, sizeof(height_d), (void*)&height_d);
	//err |= clSetKernelArg(kernel, j++, sizeof(cam_pos_d), (void*)&cam_pos_d);
	err |= clSetKernelArg(kernel, j++, sizeof(cam_pos_inv_d), (void*)&cam_pos_inv_d);
	err |= clSetKernelArg(kernel, j++, sizeof(dmap), (void*)&dmap);
	err |= clSetKernelArg(kernel, j++, sizeof(k_d), (void*)&k_d);
	err |= clSetKernelArg(kernel, j++, sizeof(fl_d), (void*)&fl_d);
	err |= clSetKernelArg(kernel, j++, sizeof(tsdf_params), (void*)&tsdf_params);

	assert(err == CL_SUCCESS);

	size_t global_ws[2] = { RESX, RESY };

#ifdef PROFILING
	LARGE_INTEGER g_PerfFrequency; 
	LARGE_INTEGER g_PerformanceCountNDRangeStart; 
	LARGE_INTEGER g_PerformanceCountNDRangeStop; 
	QueryPerformanceFrequency(&g_PerfFrequency); 
	QueryPerformanceCounter(&g_PerformanceCountNDRangeStart);
#endif

	err = clEnqueueNDRangeKernel(GpuManager::queue, kernel, 2, 0, global_ws, NULL, 0, NULL, NULL);
	clFinish(GpuManager::queue);

#ifdef PROFILING
	QueryPerformanceCounter(&g_PerformanceCountNDRangeStop);  
	float seconds = (float)((g_PerformanceCountNDRangeStop.QuadPart - g_PerformanceCountNDRangeStart.QuadPart)/(float)g_PerfFrequency.QuadPart); 
#endif

	assert(err == CL_SUCCESS);

	//clReleaseMemObject(cam_pos_d);
	err = clReleaseMemObject(cam_pos_inv_d);
	assert(err == CL_SUCCESS);
	err = clReleaseMemObject(k_d);
	assert(err == CL_SUCCESS);
}

void UpdateReconstructionModule::processData()
{
	current_frame++;
	
	if(enable)
		TSDF();

	/*UROut* packet = new UROut(TSDF_d, cam_pos_inv);
	out = DataPacket<UROut>(packet);*/

	cl_short flex = 1000;

	// count number of valid points in the tsdf
	if(MiroirManager::getShowMode() == ShowMode::TSDF)
	{
		cl_short2* tsdf_h = new cl_short2[(int)(RESX*RESY*RESZ)];
		clEnqueueReadBuffer(GpuManager::queue, TSDF_d, CL_TRUE, 0, sizeof(cl_short2)*RESX*RESY*RESZ, tsdf_h, 0, NULL, NULL);

		//boost::mutex::scoped_lock lock(MiroirManager::getMutex());
		int nPoints = 0;
		for(int i=0; i<(RESX*RESY*RESZ); i++)
		{
			cl_short tsdf = tsdf_h[i].s[0];
			if(tsdf == SHORT_NAN || tsdf > flex || tsdf < -flex)
			{
				continue;
			}
			nPoints++;
				
		}

		pcl::OpenNIGrabber* grabber = static_cast<pcl::OpenNIGrabber*>(MiroirManager::getGrabber());
		cloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud <pcl::PointXYZRGBA>);
		cloud->height = grabber->getDevice()->getDepthOutputMode().nYRes;
		cloud->width = grabber->getDevice()->getDepthOutputMode().nXRes;
		cloud->is_dense = false;

		cloud->points.resize(nPoints);

		/*if (grabber->getDevice()->isDepthRegistered ())
			cloud->header.frame_id = "/openni_rgb_optical_frame";
		else
			cloud->header.frame_id = "/openni_depth_optical_frame";*/

		int count = 0;
		
		for(int j=0; j<RESY; j++)
		{
			for(int i=0; i<RESX; i++)
			{
				for(int k=0; k<RESZ; k++)
				{
					int idx = k+i*RESZ+j*RESX*RESZ;
					
					cl_short tsdf_val = tsdf_h[idx].s[0];
					if(count >= nPoints || tsdf_val == SHORT_NAN || tsdf_val > flex || tsdf_val < -flex )
					{
						//pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN ();
						continue;
					}
					

					pcl::PointXYZRGBA& pt = cloud->points[count++];
					pt.x = 0.001 * (i * VOLUME/RESX - VOLUME/2 );
					pt.y = 0.001 * (j * VOLUME/RESY - VOLUME/2 );
					/*pt.x = 0.001 * i * VOLUME/RESX ;
					pt.y = 0.001 * j * VOLUME/RESY;*/
					pt.z = 0.001 * k * VOLUME/RESZ;
					pt.r = 255;
					pt.g = tsdf_val<0?0:255;
					pt.b = pt.g;
				}

			}
		}

		delete [] tsdf_h;

		cloud->sensor_origin_.setZero ();
		cloud->sensor_orientation_.w () = 0.0f;
		cloud->sensor_orientation_.x () = 1.0f;
		cloud->sensor_orientation_.y () = 0.0f;
		cloud->sensor_orientation_.z () = 0.0f;  
	}
	/*else
		tsdf_h = NULL;*/
	
}