#include "tracking_module.h"
#include "gpu_manager.h"
#include "miroir_manager.h"
#include "util.h"

#include <pcl/io/openni_grabber.h>

TrackingModule::TrackingModule()
{
	cl_float4 c1;
	cl_float4 c2;
	cl_float4 c3;
	cl_float4 c4;

	c1.s[0] = c2.s[1] = c3.s[2] = c4.s[3] = 1.0f;
	c1.s[1] = c1.s[2] = c1.s[3] = c2.s[0] = c2.s[2] = c2.s[3] = c3.s[0] = c3.s[1] = c3.s[3] = c4.s[0] = c4.s[1] = c4.s[2] = 0.0f;

	cam_pose[0] = c1;
	cam_pose[1] = c2;
	cam_pose[2] = c3;
	cam_pose[3] = c4;

	find_corresp = GpuManager::createKernel("../src/OpenCL/tracking_correspondence.cl", "find_correspondences");
	compute_block_sum = GpuManager::createKernel("../src/OpenCL/tracking_sum.cl", "compute_block_sums");
	compute_sum = GpuManager::createKernel("../src/OpenCL/tracking_sum.cl", "compute_sums");

	corresp_h = new cl_float2[NPIX];

	for(int i = 0; i < NPIX; i++)
	{
		corresp_h[i].s[0] = corresp_h[i].s[1] = std::numeric_limits<float>::quiet_NaN ();
	}

	cl_int err;
	corresp_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float2)*NPIX, corresp_h, &err);
	assert(err == CL_SUCCESS);

	delete [] corresp_h;
}

//#define PROFILING

void TrackingModule::findCorrespondences(const cl_float4 cam_pose_inv[4])
{
	cl_int err;

	cl_mem cam_pose_inv_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float4)*4, (void*)cam_pose_inv, &err);
	assert(err == CL_SUCCESS);

	// test
	unsigned int c=0;
	cl_mem counter = clCreateBuffer(GpuManager::context, CL_MEM_READ_WRITE, sizeof(cl_uint), 0, &err);
	assert(err == CL_SUCCESS);
	err = clEnqueueWriteBuffer(GpuManager::queue, counter, true, 0, sizeof(cl_int), &c, 0, NULL, NULL);
	assert(err == CL_SUCCESS);
	//

	cl_mem width_d = MiroirManager::getCLWidth();
	cl_mem height_d = MiroirManager::getCLHeight();
	cl_mem fl_d = MiroirManager::getCLFocalLength();

	cl_uint j = 0;
	err = clSetKernelArg(find_corresp, j++, sizeof(cl_mem), (void*)&width_d);
	err |= clSetKernelArg(find_corresp, j++, sizeof(cl_mem), (void*)&height_d);
	err |= clSetKernelArg(find_corresp, j++, sizeof(cl_mem), (void*)&fl_d);
	err |= clSetKernelArg(find_corresp, j++, sizeof(cl_mem), (void*)&cam_pose_inv_d);
	err |= clSetKernelArg(find_corresp, j++, sizeof(cl_mem), (void*)&vmap_raycast);
	err |= clSetKernelArg(find_corresp, j++, sizeof(cl_mem), (void*)&vmap_sensor);
	err |= clSetKernelArg(find_corresp, j++, sizeof(cl_mem), (void*)&nmap_raycast);
	err |= clSetKernelArg(find_corresp, j++, sizeof(cl_mem), (void*)&nmap_sensor);
	err = clSetKernelArg(find_corresp, j++, sizeof(cl_mem), (void*)&corresp_d);
	clSetKernelArg(find_corresp, j++, sizeof(counter), &counter);

	assert(err == CL_SUCCESS);

	int width, height;
	MiroirManager::getSensorDimensions(width, height);

	size_t global_ws[2] = { width, height };
	//size_t local_ws[2] = { 16, 12 };

#ifdef PROFILING
	LARGE_INTEGER g_PerfFrequency; 
	LARGE_INTEGER g_PerformanceCountNDRangeStart; 
	LARGE_INTEGER g_PerformanceCountNDRangeStop; 
	QueryPerformanceFrequency(&g_PerfFrequency); 
	QueryPerformanceCounter(&g_PerformanceCountNDRangeStart);
#endif

	err = clEnqueueNDRangeKernel(GpuManager::queue, find_corresp, 2, 0, global_ws, NULL, 0, NULL, NULL);
	clFinish(GpuManager::queue);
	
#ifdef PROFILING
	QueryPerformanceCounter(&g_PerformanceCountNDRangeStop);  
	float seconds = (float)((g_PerformanceCountNDRangeStop.QuadPart - g_PerformanceCountNDRangeStart.QuadPart)/(float)g_PerfFrequency.QuadPart); 
#endif

	assert(err == CL_SUCCESS);

	if(MiroirManager::getShowMode() == ShowMode::CORRESP)
	{
		clEnqueueReadBuffer(GpuManager::queue, counter, CL_TRUE, 0, sizeof(cl_uint), &corresp_count, 0, NULL, NULL);
		corresp_h = new cl_float2[NPIX];
		clEnqueueReadBuffer(GpuManager::queue, corresp_d, CL_TRUE, 0, sizeof(cl_float2)*NPIX, corresp_h, 0, NULL, NULL);	
	}
	else
		corresp_h = NULL;

	clReleaseMemObject(cam_pose_inv_d);
}

#define WORK_GROUP_SIZE 64
#define NUM_WORK_GROUP NPIX /2 / WORK_GROUP_SIZE
#define A_SIZE 21
#define B_SIZE 6

// TODO fix memory leak here
void TrackingModule::sumLinearSystem(cl_float A_h[36], cl_float B_h[6], const cl_float4 est_pose[4])
{
	cl_int err;
	
	// step 1 : calculate the summands and get first intermediate sums
	cl_mem A_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_WRITE, sizeof(cl_float)*NUM_WORK_GROUP*A_SIZE, NULL, &err);
	assert(err == CL_SUCCESS);
	cl_mem B_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_WRITE, sizeof(cl_float)*NUM_WORK_GROUP*B_SIZE, NULL, &err);
	assert(err == CL_SUCCESS);
	cl_mem est_pose_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float4)*4, (void*)est_pose, &err);
	assert(err == CL_SUCCESS);

	cl_mem width_d = MiroirManager::getCLWidth();
	cl_mem height_d = MiroirManager::getCLHeight();
	cl_mem fl_d = MiroirManager::getCLFocalLength();

	cl_uint j = 0;
	err = clSetKernelArg(compute_block_sum, j++, sizeof(cl_mem), (void*)&width_d);
	err |= clSetKernelArg(compute_block_sum, j++, sizeof(cl_mem), (void*)&est_pose_d);
	err |= clSetKernelArg(compute_block_sum, j++, sizeof(cl_mem), (void*)&corresp_d);
	err |= clSetKernelArg(compute_block_sum, j++, sizeof(cl_mem), (void*)&vmap_sensor);
	err |= clSetKernelArg(compute_block_sum, j++, sizeof(cl_mem), (void*)&vmap_raycast);
	err |= clSetKernelArg(compute_block_sum, j++, sizeof(cl_mem), (void*)&nmap_raycast);
	err |= clSetKernelArg(compute_block_sum, j++, sizeof(cl_mem), (void*)&A_d);
	err |= clSetKernelArg(compute_block_sum, j++, sizeof(cl_mem), (void*)&B_d);
	assert(err == CL_SUCCESS);

	int width, height;
	MiroirManager::getSensorDimensions(width, height);

	size_t global_ws[1] = { width * height / 2 };	// each thread sums 2 elements
	size_t local_ws[1] = { WORK_GROUP_SIZE };

#ifdef PROFILING
	LARGE_INTEGER g_PerfFrequency; 
	LARGE_INTEGER g_PerformanceCountNDRangeStart; 
	LARGE_INTEGER g_PerformanceCountNDRangeStop; 
	QueryPerformanceFrequency(&g_PerfFrequency); 
	QueryPerformanceCounter(&g_PerformanceCountNDRangeStart);
#endif
	err = clEnqueueNDRangeKernel(GpuManager::queue, compute_block_sum, 1, 0, global_ws, local_ws, 0, NULL, NULL);
	clFinish(GpuManager::queue);
#ifdef PROFILING
	QueryPerformanceCounter(&g_PerformanceCountNDRangeStop);  
	float seconds = (float)((g_PerformanceCountNDRangeStop.QuadPart - g_PerformanceCountNDRangeStart.QuadPart)/(float)g_PerfFrequency.QuadPart); 
#endif

	assert(err == CL_SUCCESS);

//	cl_float3* test = new cl_float3[307200];
//	cl_float* test = new cl_float[NUM_WORK_GROUP*B_SIZE];
//	clEnqueueReadBuffer(GpuManager::queue, B_d, CL_TRUE, 0, sizeof(cl_float)*NUM_WORK_GROUP*B_SIZE, test, 0, NULL, NULL);
//	clEnqueueReadBuffer(GpuManager::queue, vmap_raycast, CL_TRUE, 0, sizeof(cl_float3)*307200, test, 0, NULL, NULL);
	
	clFinish(GpuManager::queue);

	// step 2 : sum intermediate sums
	j = 0;
	int size = (float)global_ws[0] / (float)local_ws[0] ;
	int nblocks = std::ceil( size / 2.0f / (float)WORK_GROUP_SIZE);
	int nitems = nblocks * WORK_GROUP_SIZE;
	cl_mem A2_d = clCreateBuffer(GpuManager::context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * nblocks * A_SIZE, NULL, &err);
	assert(err == CL_SUCCESS);
	cl_mem B2_d = clCreateBuffer(GpuManager::context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * nblocks * B_SIZE, NULL, &err);
	assert(err == CL_SUCCESS);
	cl_mem size_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &size, &err);
	assert(err == CL_SUCCESS);

	err = clSetKernelArg(compute_sum, j++, sizeof(cl_mem), (void*)&size_d);
	err |= clSetKernelArg(compute_sum, j++, sizeof(cl_mem), (void*)&A_d);
	err |= clSetKernelArg(compute_sum, j++, sizeof(cl_mem), (void*)&B_d);
	err |= clSetKernelArg(compute_sum, j++, sizeof(cl_mem), (void*)&A2_d);
	err |= clSetKernelArg(compute_sum, j++, sizeof(cl_mem), (void*)&B2_d);
	assert(err == CL_SUCCESS);

	global_ws[0] = nitems;	
	local_ws[0] = WORK_GROUP_SIZE;	

#ifdef PROFILING
	QueryPerformanceFrequency(&g_PerfFrequency); 
	QueryPerformanceCounter(&g_PerformanceCountNDRangeStart);
#endif
	err = clEnqueueNDRangeKernel(GpuManager::queue, compute_sum, 1, 0, global_ws, local_ws, 0, NULL, NULL);
	clFinish(GpuManager::queue);
#ifdef PROFILING
	QueryPerformanceCounter(&g_PerformanceCountNDRangeStop);  
	seconds = (float)((g_PerformanceCountNDRangeStop.QuadPart - g_PerformanceCountNDRangeStart.QuadPart)/(float)g_PerfFrequency.QuadPart); 
#endif

	assert(err == CL_SUCCESS);

	cl_float* ABlocks = new cl_float[nblocks*A_SIZE];
	cl_float* BBlocks = new cl_float[nblocks*B_SIZE];
	clEnqueueReadBuffer(GpuManager::queue, A2_d, CL_TRUE, 0, sizeof(cl_float) * nblocks * A_SIZE, ABlocks, 0, NULL, NULL);
	clEnqueueReadBuffer(GpuManager::queue, B2_d, CL_TRUE, 0, sizeof(cl_float) * nblocks * B_SIZE, BBlocks, 0, NULL, NULL);

	clFinish(GpuManager::queue);

	// get final A and B
	int col = 0;
	int pad = 0;
	int buf = 0;
	for(int i = 0; i < nblocks * A_SIZE; i++)
	{
		A_h[i%A_SIZE+pad] += ABlocks[i];
		if((i+1 - buf) % (6 - col) == 0){ col++; pad += col; buf = i+1; }
		if(col == 6){ col = 0; pad = 0; }
	}
	for(int i = 0; i < nblocks * B_SIZE; i++)
		B_h[i%B_SIZE] += BBlocks[i];

	clReleaseMemObject(A_d);
	clReleaseMemObject(B_d);
	clReleaseMemObject(A2_d);
	clReleaseMemObject(B2_d);
	clReleaseMemObject(est_pose_d);
	clReleaseMemObject(size_d);
	delete [] ABlocks;
	delete [] BBlocks;
}

void TrackingModule::processData()
{
	cl_float4* est_pose = new cl_float4[4];
	cl_float4* inc_trans = new cl_float4[4];
	cl_float4* pose_inv = new cl_float4[4];
	cl_float4* cam_pose_prev = new cl_float4[4];
	cl_float x[6];

	est_pose[0] = cam_pose[0];
	est_pose[1] = cam_pose[1];
	est_pose[2] = cam_pose[2];
	est_pose[3] = cam_pose[3];

	MiroirUtil::invertCamTransform(cam_pose, pose_inv);
	MiroirUtil::mat4Mult(pose_inv, est_pose, inc_trans);

	// 1. for each pixel find vertex correspondences
	findCorrespondences(pose_inv);

	for(int i=0; i<15; i++)
	{
		
		// 2. sum matrix A and vector b of the linear system 
		cl_float A_h[36], B_h[6];
		cl_float decomp[36];
		// init A_h and B_h
		for(int k = 0; k < 36; k++)		
			decomp[k] = A_h[k] = 0.0f;
		for(int k = 0; k < 6; k++)
			B_h[k] = 0.0f;
		for(int k = 0; k < 6; k++)
			A_h[k*6+k] = 1.0f;

		sumLinearSystem(A_h, B_h, est_pose);	

		for(int k = 0; k < 6; k++)
		{
			for(int j = k+1; j<6; j++)
				A_h[j * 6 + k] = A_h[k * 6 + j];
		}

		// checking nullspace
		double det = MiroirUtil::determinant(A_h, 6);
		if(std::fabs(det) < 1e-15 || det != det)
		{
			// lost tracking
			break;
		}

		MiroirUtil::cholesky(6, A_h, decomp);

		// 3. find solution of linear system on CPU using Cholesky decomposition
		// L*y = b, y = trans(L)*x
		cl_float y[6];
		for(int k = 0; k < 6; k++) 
			x[k] = y[k] = 0;
		for(int k = 0; k < 6; k++)
		{
			y[k] = B_h[k] / decomp[k * 6 + k];
			for(int j = 0; j < k; j++)
				y[k] -= decomp[k * 6 + j] * y[j] / decomp[k * 6 + k];
		}
		for(int k = 5; k >= 0; k--)
		{
			x[k] = y[k] / decomp[k * 6 + k];
			for(int j = 0; j < k; j++)
				x[k] -= decomp[j * 6 + k] * x[k] / decomp[k * 6 + k];
		}

		if(std::sqrt(x[3]*x[3]+x[4]*x[4]+x[5]*x[5]) > 100)	// lost tracking
			break;

		cam_pose_prev[0] = est_pose[0];
		cam_pose_prev[1] = est_pose[1];
		cam_pose_prev[2] = est_pose[2];
		cam_pose_prev[3] = est_pose[3];

		inc_trans[0].s[0] = inc_trans[1].s[1] = inc_trans[2].s[2] = 1.0f; 
		inc_trans[0].s[1] = -x[2];
		inc_trans[0].s[2] = x[1];
		inc_trans[1].s[0] = x[2];
		inc_trans[1].s[2] = -x[0];
		inc_trans[2].s[0] = -x[1];
		inc_trans[2].s[1] = x[0];

		inc_trans[3].s[0] = x[3];
		inc_trans[3].s[1] = x[4];
		inc_trans[3].s[2] = x[5];

		MiroirUtil::mat4Mult(inc_trans, cam_pose_prev, est_pose);
	}

	cam_pose[0] = est_pose[0];
	cam_pose[1] = est_pose[1];
	cam_pose[2] = est_pose[2];
	cam_pose[3] = est_pose[3];

	delete [] pose_inv;
	delete [] cam_pose_prev;

	if(corresp_h !=	NULL)
	{
		boost::mutex::scoped_lock lock(MiroirManager::getMutex());
		cl_float3* vmap_raycast_h = new cl_float3[NPIX];
		cl_float3* vmap_sensor_h = new cl_float3[NPIX];
		clEnqueueReadBuffer(GpuManager::queue, vmap_raycast, CL_TRUE, 0, sizeof(cl_float3)*NPIX, vmap_raycast_h, 0, NULL, NULL);
		clEnqueueReadBuffer(GpuManager::queue, vmap_sensor, CL_TRUE, 0, sizeof(cl_float3)*NPIX, vmap_sensor_h, 0, NULL, NULL);
		clFinish(GpuManager::queue);

		pcl::OpenNIGrabber* grabber = static_cast<pcl::OpenNIGrabber*>(MiroirManager::getGrabber());
		cloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud <pcl::PointXYZRGBA>);
		cloud->height = grabber->getDevice()->getDepthOutputMode().nYRes;
		cloud->width = grabber->getDevice()->getDepthOutputMode().nXRes;
		cloud->is_dense = false;
		cloud->points.resize(corresp_count*2);

		int width, height;
		MiroirManager::getSensorDimensions(width, height);

		int count = 0;
		for(int i = 0; i < NPIX; i++)
		{
			// if is nan
			if(corresp_h[i].s[0] != corresp_h[i].s[0])
				continue;

			pcl::PointXYZRGBA& pt = cloud->points[count++];
			int idx = width * corresp_h[i].s[1] + corresp_h[i].s[0];
			pt.x = 0.001 * vmap_raycast_h[idx].s[0];
			pt.y = 0.001 * vmap_raycast_h[idx].s[1];
			/*pt.x = 0.001 * i * VOLUME/RESX ;
			pt.y = 0.001 * j * VOLUME/RESY;*/
			pt.z = 0.001 * vmap_raycast_h[idx].s[2];
			pt.r = 127;
			pt.g = 0;
			pt.b = pt.g;

			pcl::PointXYZRGBA& pt2 = cloud->points[count++];
			
			pt2.x = 0.001 * vmap_sensor_h[i].s[0];
			pt2.y = 0.001 * vmap_sensor_h[i].s[1];
			/*pt.x = 0.001 * i * VOLUME/RESX ;
			pt.y = 0.001 * j * VOLUME/RESY;*/
			pt2.z = 0.001 * vmap_sensor_h[i].s[2];
			pt2.r = 0;
			pt2.g = 0;
			pt2.b = 127;
		}

		delete [] corresp_h;
		delete [] vmap_raycast_h;
		delete [] vmap_sensor_h;
	}
}