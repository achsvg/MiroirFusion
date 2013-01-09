#ifndef RAYCAST_MODULE_H
#define RAYCAST_MODULE_H

#include "definitions.h"
#include "miroir_module.h"
#include "window_listener.h"

typedef int RaycastOut;

class RaycastModule : public MiroirModule, public MiroirWindowListener
{
private:
	boost::mutex raycast_mutex;

	cl_kernel raycast_kernel;

	cl_float3* vmap_h;
	cl_float3* nmap_h;

	cl_mem vmap_d;
	cl_mem nmap_d;
	cl_mem tex_d;
	unsigned int gl_tex;

	int tex_dim[2];

	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloud;

	// to be set by tsdf module
	cl_mem tsdf;
	cl_float4 cam_pose[4];
	//cl_float* tsdf_h;

public:
	RaycastModule();
	~RaycastModule();

	void raycast(cl_mem TSDF, cl_float4 cam_pos[4]);

	void processData();

	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr getCloud() const { return cloud; }

	void onWindowResized(int width, int height);

	void setTSDF(const cl_mem& buf)
	{
		tsdf = buf;
	}

	void setCamPose(const cl_float4 pose[4])
	{
		cam_pose[0] = pose[0];
		cam_pose[1] = pose[1];
		cam_pose[2] = pose[2];
		cam_pose[3] = pose[3];
	}

	cl_mem getVertexMap() const
	{
		return vmap_d;
	}

	cl_mem getNormalMap() const
	{
		return nmap_d;
	}
};

#endif