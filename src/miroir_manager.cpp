#include "miroir_manager.h"

#include "miroir_module_manager.h"
#include "tracking_module.h"
#include "surface_measurement_module.h"
#include "update_reconstruction_module.h"
#include "raycast_module.h"
#include "gpu_manager.h"
#include "../src/OpenCL/gpu_def.h"
#include "window_listener.h"
#include "profiler.h"
#include "util.h"

#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/oni_grabber.h>
#include <pcl/visualization/pcl_visualizer.h>

//#define PROFILER

MiroirManager& MiroirManager::miroir = MiroirManager();
std::vector<MiroirWindowListener*> MiroirManager::windowListeners;
boost::mutex MiroirManager::miroir_mutex;

MiroirManager::MiroirManager() : mode(DEPTH_MAP_BIL), frame(0)
{
	// setting up the point cloud grabber for Kinect device
	grabber = new pcl::OpenNIGrabber();
	//grabber = new pcl::ONIGrabber("C:/Program Files (x86)/OpenNI/Samples/Bin/Release/20120808-133730.oni",true,false);

	boost::function<void (const boost::shared_ptr<openni_wrapper::DepthImage>&)> callback = 
		boost::bind(&MiroirManager::setCurrentDepthImage, this, _1);	
	grabber->registerCallback(callback);

	boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f =
         boost::bind (&MiroirManager::setRawCloud, this, _1);

	grabber->registerCallback(f);

	boost::shared_ptr< openni_wrapper::OpenNIDevice > device = static_cast<pcl::OpenNIGrabber*>(grabber)->getDevice();
	sensor_width = device->getDefaultDepthMode().nXRes;
	sensor_height = device->getDefaultDepthMode().nYRes;
	focal_length = device->getDepthFocalLength();
	principal_point.s[0] = 320.0f;//width * 0.5f;
	principal_point.s[1] = 240.0f;//height * 0.5f;
}

MiroirManager::~MiroirManager()
{
	//delete grabber;
	delete surf_measurement;
	delete reconstruct;
	delete rayc;
	clReleaseMemObject(width_d);
	clReleaseMemObject(height_d);
	clReleaseMemObject(fl_d);
}

void MiroirManager::setRunTSDF(bool enable)
{
	MiroirManager::getSingleton().reconstruct->setEnable(enable);
}

bool MiroirManager::isTSDFRunning() 
{ 
	return MiroirManager::getSingleton().reconstruct->isEnabled(); 
}

void MiroirManager::buildMiroir()
{
	//SurfaceMeasurementModulePtr
	surf_measurement = new SurfaceMeasurementModule(sensor_width, sensor_height, focal_length);
	reconstruct = new UpdateReconstructionModule();
	rayc = new RaycastModule();
	pose_estimation = new TrackingModule();
	MiroirModuleManager::getSingleton().plug(*surf_measurement, *pose_estimation);
}

void MiroirManager::initGlobalBuffers()
{
	cl_int err;
	TsdfParams params;
	params.resolution[0] = RESX;
	params.resolution[1] = RESY;
	params.resolution[2] = RESZ;
	params.volume[0] = VOLUMEX;
	params.volume[1] = VOLUMEY;
	params.volume[2] = VOLUMEZ;
	params.mu = MU;

	miroir.width_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &miroir.sensor_width, &err);
	assert(err == CL_SUCCESS);
	miroir.height_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &miroir.sensor_height, &err);
	assert(err == CL_SUCCESS);
	miroir.fl_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float), &miroir.focal_length, &err);
	assert(err == CL_SUCCESS);
	miroir.tsdf_params_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(TsdfParams), &params, &err);
	assert(err == CL_SUCCESS);
}

int bli = 0;

void MiroirManager::step()
{
	if( depth_image == NULL )
		return;

	cl_int err;
	cl_mem depth_map_d = clCreateBuffer(GpuManager::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned short)*NPIX, (void*)(depth_image->getDepthMetaData().Data()), &err);
	assert(err == CL_SUCCESS);

	/// begin Test 
	/*cl_mem prev_surf = clCreateBuffer(GpuManager::context, CL_MEM_READ_WRITE, sizeof(cl_float3)*NPIX, 0, &err);
	if(bli != 0)
		clEnqueueCopyBuffer(GpuManager::queue, surf_measurement->getVertexMap(), prev_surf, 0, 0, sizeof(cl_float3)*NPIX, 0, 0, 0);*/
	/// end Test
	clFinish(GpuManager::queue);

	surf_measurement->setDepthMap(depth_map_d);
	MiroirProfiler::start("surface measurement");
	surf_measurement->processData();
	MiroirProfiler::stop("surface measurement");

	/*if(bli == 0)
		clEnqueueCopyBuffer(GpuManager::queue, surf_measurement->getVertexMap(), prev_surf, 0, 0, sizeof(cl_float3)*NPIX, 0, 0, 0);
	clFinish(GpuManager::queue);*/

	// check if camera moved if yes integrate volume
	cl_float4 t;
	cl_float4* camprev = reconstruct->getCamPose();
	cl_float4* camcurr = pose_estimation->getCamPose();
	cl_float4 camcurr_inv[4];
	cl_float3 rot_inc[3];

	t.s[0] = camcurr[3].s[0] - camprev[3].s[0];
	t.s[1] = camcurr[3].s[1] - camprev[3].s[1];
	t.s[2] = camcurr[3].s[2] - camprev[3].s[2];
	t.s[3] = camcurr[3].s[3] - camprev[3].s[3];

	cl_float tnorm = std::sqrt(t.s[0]*t.s[0] + t.s[1]*t.s[1] + t.s[2]*t.s[2] + t.s[3]*t.s[3]);
	MiroirUtil::invertCamTransform(camcurr,camcurr_inv);
	MiroirUtil::mat3Mult(camcurr_inv, camprev, rot_inc);
	cl_float r = MiroirUtil::angleFromRotMatrix(rot_inc);
	//////////////////////////////////////////////////////////////

	//if((r > ROTATION_THRES && tnorm > TRANSLATION_THRES) || frame < 20)
	//{
		MiroirProfiler::start("tsdf");
		reconstruct->setDepthMap(depth_map_d);
		reconstruct->setCamPose(pose_estimation->getCamPose());
		reconstruct->processData();
		MiroirProfiler::stop("tsdf");
	//}

	clReleaseMemObject(depth_map_d);

	MiroirProfiler::start("raycast");
	//rayc->input( reconstruct->output() );
	rayc->setTSDF(reconstruct->getTSDF());
	rayc->setCamPose(pose_estimation->getCamPose());
	rayc->processData();
	MiroirProfiler::stop("raycast");

	MiroirProfiler::start("Pose estimation");
	pose_estimation->setParams(rayc->getVertexMap(), surf_measurement->getVertexMap(), rayc->getNormalMap(), surf_measurement->getNormalMap());
	pose_estimation->processData();
	MiroirProfiler::stop("Pose estimation");
	frame++;
//	clReleaseMemObject(prev_surf);

//	bli++;
	//MiroirModuleManager::getSingleton().feed( *surf_measurement, data );
	//MiroirModuleManager::getSingleton().step();
}

#define PROFILING

void MiroirManager::run()
{
#if defined(PROFILER)
	MiroirProfiler::enable();
#endif
	

	grabber->start();
	isRunning = true;
	while( isRunning )
	{
		MiroirProfiler::start("main loop");
		#ifdef PROFILING
			LARGE_INTEGER g_PerfFrequency; 
			LARGE_INTEGER g_PerformanceCountNDRangeStart; 
			LARGE_INTEGER g_PerformanceCountNDRangeStop; 
			QueryPerformanceFrequency(&g_PerfFrequency); 
			QueryPerformanceCounter(&g_PerformanceCountNDRangeStart);
		#endif
		step();
		#ifdef PROFILING
			QueryPerformanceCounter(&g_PerformanceCountNDRangeStop);  
			float seconds = (float)((g_PerformanceCountNDRangeStop.QuadPart - g_PerformanceCountNDRangeStart.QuadPart)/(float)g_PerfFrequency.QuadPart); 
		#endif	
		MiroirProfiler::stop("main loop");
		//std::cout << std::endl;
	}
}

void MiroirManager::setCurrentDepthImage(const boost::shared_ptr<openni_wrapper::DepthImage>& depthImg)
{
	depth_image = depthImg;
}

//void MiroirManager::setCurrentCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
//{
//	boost::mutex::scoped_lock lock(miroir_mutex);
//	current_cloud = cloud;
//}

pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr MiroirManager::getOutputCloud()
{
	boost::mutex::scoped_lock lock(miroir_mutex);
	if(mode == DEPTH_MAP_RAW)
		return raw_cloud;
	if(mode == DEPTH_MAP_BIL)
		return surf_measurement->getFilteredCloud();
	if(mode == TSDF)
		return reconstruct->getCloud();
	if(mode == RAYCAST)
		return rayc->getCloud();
	if(mode == CORRESP)
		return pose_estimation->getCloud();
	return pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr();
}

void MiroirManager::setWindowDimensions(int width, int height)
{ 
	MiroirManager::getSingleton().w_width = width; 
	MiroirManager::getSingleton().w_height = height; 

	for(auto it = windowListeners.begin(); it != windowListeners.end(); ++it)
	{
		(*it)->onWindowResized(width, height);
	}
}

void MiroirManager::getSensorDimensions(int& w, int& h)
{ 
	w = MiroirManager::getSingleton().sensor_width; 
	h = MiroirManager::getSingleton().sensor_height; 
}

void MiroirManager::getWindowDimensions(int& w, int& h)
{ 
	w = MiroirManager::getSingleton().w_width; 
	h = MiroirManager::getSingleton().w_height; 
}

float MiroirManager::getFocalLength()
{
	return MiroirManager::getSingleton().focal_length;
}

cl_float2 MiroirManager::getPrincipalPoint()
{
	return MiroirManager::getSingleton().principal_point;
}

cl_float4* MiroirManager::getCameraPose()
{
	boost::mutex::scoped_lock lock(MiroirManager::getSingleton().miroir_mutex);
	return MiroirManager::getSingleton().pose_estimation->getCamPose();
}
