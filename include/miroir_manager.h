#ifndef MIROIR_MANAGER_H
#define MIROIR_MANAGER_H

//#include <pcl/io/openni_camera/openni_depth_image.h>
#include <pcl/point_types.h>
#include "definitions.h"


/** \brief Manager for the whole application. 
  * \author Anthony Chansavang <anthony.chansavang@gmail.com>
  */

class MiroirWindowListener;

enum ShowMode {
	NONE,
	DEPTH_MAP_BIL,
	DEPTH_MAP_RAW,
	TSDF,
	RAYCAST,
	CORRESP
};

class MiroirManager
{
private:
	static MiroirManager& miroir;
	pcl::Grabber* grabber;										// frame or point cloud grabber
	pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr raw_cloud;
	boost::shared_ptr<openni_wrapper::DepthImage> depth_image;		// current point cloud
	static std::vector<MiroirWindowListener*> windowListeners;

	// modules
	SurfaceMeasurementModule* surf_measurement;
	TrackingModule* pose_estimation;
	UpdateReconstructionModule* reconstruct;
	RaycastModule* rayc;
	
	static boost::mutex miroir_mutex;

	int w_width;						// window's width
	int w_height;
	int sensor_width;					// depth sensors' width
	int sensor_height;
	float focal_length;
	cl_float2 principal_point;
	bool isRunning;
	bool runTSDF;

	unsigned long frame;

	cl_mem width_d;
	cl_mem height_d;
	cl_mem fl_d;
	cl_mem tsdf_params_d;

	ShowMode mode;

	MiroirManager();
	
	/** \brief set the current depth img, used as a callback for the grabber.
	  * \param[in] depthImg depth img.
      */
	void setCurrentDepthImage(const boost::shared_ptr<openni_wrapper::DepthImage>& depthImg);

	/** \brief set the current depth img, used as a callback for the grabber.
	  * \param[in] depthImg depth img.
      */
	void setRawCloud(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& raw)
	{
		raw_cloud = raw;
	}

public:

	~MiroirManager();
	
	inline static MiroirManager& getSingleton()
	{
		return miroir;
	}
	
	/** \brief retrieve the current point cloud.
      * \return current point cloud.
      */
	//pcl::PointCloud<pcl::PointXYZ>::Ptr getCurrentCloud()
	//{
	//	boost::mutex::scoped_lock lock(miroir_mutex);
	//	return current_cloud;
	//}
	
	pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr getOutputCloud();

	void step();

	/** \brief run the machine.
	  */
	void run();

	/** \brief assembles the different modules, using the module manager.
	  */
	void buildMiroir();

	static void stop(){ MiroirManager::miroir.isRunning = false; }

	static pcl::Grabber* getGrabber() { return MiroirManager::getSingleton().grabber; }

	void setWindowDimensions(int width, int height);
	
	static void getSensorDimensions(int& w, int& h);
	static void getWindowDimensions(int& w, int& h);
	static float getFocalLength();
	static cl_float2 getPrincipalPoint();

	static cl_mem getCLWidth(){ return MiroirManager::getSingleton().width_d; }
	static cl_mem getCLHeight(){ return MiroirManager::getSingleton().height_d; }
	static cl_mem getCLFocalLength(){ return MiroirManager::getSingleton().fl_d; }
	static cl_mem getCLTSDFParams(){ return MiroirManager::getSingleton().tsdf_params_d; }

	static void setShowMode(ShowMode smode){ MiroirManager::getSingleton().mode = smode; }
	static ShowMode getShowMode(){ return MiroirManager::getSingleton().mode; }

	static void initGlobalBuffers();

	static void addWindowListener(MiroirWindowListener* listener){ windowListeners.push_back(listener); }

	static cl_float4* getCameraPose();

	static boost::mutex& getMutex() { return miroir_mutex; }

	static void setRunTSDF(bool enable);

	static bool isTSDFRunning();
};

#endif