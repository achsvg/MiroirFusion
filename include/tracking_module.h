#ifndef SENSOR_POSE_MODULE_H
#define SENSOR_POSE_MODULE_H

#include "definitions.h"
#include "miroir_module.h"

class TrackingModule : public MiroirModule
{
private:
	cl_mem vmap_raycast;
	cl_mem vmap_sensor;
	cl_mem nmap_raycast;
	cl_mem nmap_sensor;
	cl_mem corresp_d;		// buffer for found correspondences 
	cl_float4 cam_pose[4]; 

	cl_float2* corresp_h;
	cl_uint corresp_count;

	cl_kernel find_corresp;
	cl_kernel compute_sum;
	cl_kernel compute_block_sum;

	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloud;

	void findCorrespondences(const cl_float4 cam_pose_inv[4]);
	void sumLinearSystem(cl_float A_h[36], cl_float B_h[6], const cl_float4 cam_est_pose[4]);
public:
	TrackingModule();

	void processData();

	inline void setParams(const cl_mem& vraycast, const cl_mem& vsensor, const cl_mem& nraycast, const cl_mem& nsensor)
	{
		vmap_raycast = vraycast;
		vmap_sensor = vsensor;
		nmap_raycast = nraycast;
		nmap_sensor = nsensor;
	}

	inline cl_float4* getCamPose() const
	{
		return (cl_float4*)cam_pose;
	}

	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr getCloud() const { return cloud; }
};

//typedef std::shared_ptr< TrackingModule > TrackingModulePtr;

#endif