#ifndef SURFACE_MEASUREMENT_MODULE_H
#define SURFACE_MEASUREMENT_MODULE_H

#include "definitions.h"

#include "miroir_module.h"

#include <pcl/io/openni_camera/openni_depth_image.h>
#include <pcl/point_types.h>

class SurfaceMeasurementModule : public MiroirModule
{
private:
	unsigned short* dmap_h;
	cl_float3* vmap_h;
	cl_float3* nmap_h;

	cl_mem vmap_d;
	cl_mem nmap_d;

	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloud_filtered;

	cl_kernel measurement_vertices;
	cl_kernel measurement_normals;

	void bilateralFilter( cl_mem raw_dmap );

	cl_mem sigma_s_d;
	cl_mem sigma_r_d;

	// to be set
	cl_mem dmap;
public:

	SurfaceMeasurementModule( int w, int h, float focal_length );

	~SurfaceMeasurementModule()
	{
		clReleaseKernel(measurement_vertices);
		clReleaseKernel(measurement_normals);
		clReleaseMemObject(sigma_s_d);
		clReleaseMemObject(sigma_r_d);
		clReleaseMemObject(vmap_d);
		clReleaseMemObject(nmap_d);
		delete [] dmap_h;
		delete [] vmap_h;
		delete [] nmap_h;
	}

	void processData();

	pcl::PointCloud< pcl::PointXYZRGBA >::ConstPtr getFilteredCloud()
	{
		return cloud_filtered;
	}

	inline void setDepthMap( const cl_mem& depth_map )
	{
		dmap = depth_map;
	}

	inline cl_mem getVertexMap() const
	{
		return vmap_d;
	}

	inline cl_mem getNormalMap() const
	{
		return nmap_d;
	}
};

//typedef std::shared_ptr< SurfaceMeasurementModule > SurfaceMeasurementModulePtr;

#endif