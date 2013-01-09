#ifndef UPDATE_RECONSTRUCTION_MODULE_H
#define UPDATE_RECONSTRUCTION_MODULE_H

#include "definitions.h"
#include "miroir_module.h"

namespace openni_wrapper
{
	class DepthImage;
}



class UpdateReconstructionModule : public MiroirModule
{
private:
	cl_kernel kernel;
	cl_ulong current_frame;
	cl_mem TSDF_d;
	cl_float4* cam_pose;
	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloud;

	cl_mem dmap;

	bool enable;

public:

	UpdateReconstructionModule();
	~UpdateReconstructionModule();

	void TSDF();

	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr getCloud() const { return cloud; }

	void processData();

	inline void setCamPose( const cl_float4 pose[4] )
	{
		cam_pose[0] = pose[0];
		cam_pose[1] = pose[1];
		cam_pose[2] = pose[2];
		cam_pose[3] = pose[3];
	}

	inline void setDepthMap( const cl_mem& depth_map )
	{
		dmap = depth_map;
	}

	inline cl_mem getTSDF() const
	{
		return TSDF_d;
	}

	inline cl_float4* getCamPose() const
	{
		return cam_pose;
	}

	inline void setEnable(bool b)
	{
		enable = b;
	}

	inline bool isEnabled() const
	{
		return enable;
	}
};

#endif