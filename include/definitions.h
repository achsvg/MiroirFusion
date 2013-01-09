#ifndef DEFINITIONS_H
#define DEFINITIONS_H

// TODO : make this into a precompiled header

#include <CL/cl.h>
#include <memory>
#include <boost/shared_ptr.hpp>
#include <boost/thread/thread.hpp>
#include <pcl/point_cloud.h>

//#include "surface_measurement_module.h"

#define RESX 512.0f
#define RESY 512.0f
#define RESZ 512.0f

#define VOLUMEX 1500.0f	// volume of scanned area in mm
#define VOLUMEY 1500.0f
#define VOLUMEZ 1500.0f

#define SENSOR_ZOFFSET 400.0f	// blind spot in mm

#define NPIX 307200

#define MU 30.0f		// truncated distance in mm

#define VOLUME VOLUMEX

#define ROTATION_THRES 0.00314159265359f		// in rad
#define TRANSLATION_THRES 5						// in mm

namespace pcl
{
	struct PointXYZ;
	struct PointXYZRGBA;
	class Grabber;
}

namespace openni_wrapper
{
	class DepthImage;
}

struct MemObj
{
	cl_mem o;
};

class SurfaceMeasurementModule;
class TrackingModule;
class UpdateReconstructionModule;
class RaycastModule;

//typedef std::tuple< cl_mem, cl_float4** > RaycastIn;	// <tsdf, cam pose>

//typedef RaycastIn UROut;
typedef cl_float3* VertexMap;
typedef cl_float3* NormalMap;

//typedef MemObj SurfaceMeasurementIn;	// mem obj for the depth map
//typedef SurfaceMeasurementIn URIn;
//typedef std::tuple< VertexMap, NormalMap > SurfaceMeasurementOut;
//
//typedef std::tuple< VertexMap, VertexMap, NormalMap, NormalMap > PoseEstimationIn; 
//typedef int PoseEstimationOut; /* TODO : here it is a matrix */

// explicit instantiation declaration of templates
// extern template MiroirModule<> ...

#endif