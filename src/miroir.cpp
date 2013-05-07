
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <boost/thread/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <pcl/point_types.h>

#define GLFW_NO_GLU

#include <GL/glew.h>
#include <GL/glfw.h>

#include "miroir_manager.h"
#include "gpu_manager.h"

#include <pcl/io/openni_grabber.h>
#include <pcl/io/oni_grabber.h>
//#ifdef _WIN32
//	#define sleep(x) Sleep((x)*1000)
//#endif

bool stop = false;
GLuint vao;
GLuint quad_vertexbuffer;

void GLFWCALL onWindowResized( int width, int height )
{
	MiroirManager::getSingleton().setWindowDimensions(width, height);
}

void keyboardCallback(const pcl::visualization::KeyboardEvent &evt, void *)
{
	if(evt.getKeySym() == "F1")
    {
		MiroirManager::setShowMode(DEPTH_MAP_RAW);
	}
	else if(evt.getKeySym() == "F2")
	{
		MiroirManager::setShowMode(DEPTH_MAP_BIL);
	}
	else if(evt.getKeySym() == "F3")
	{
		MiroirManager::setShowMode(TSDF);	
	}
	else if(evt.getKeySym() == "F4")
	{
		MiroirManager::setShowMode(RAYCAST);
	}
	else if(evt.getKeySym() == "F5")
	{
		MiroirManager::setShowMode(CORRESP);
	}
	else if(evt.getKeySym() == "t" && evt.keyDown())
	{
		MiroirManager::setRunTSDF(!MiroirManager::isTSDFRunning());
	}
	else if(evt.getKeySym() == "a" && evt.keyDown())
	{
		//MiroirManager::setRunTSDF(!MiroirManager::isTSDFRunning());
	}
}

void GLFWCALL keyboardCallback(int key, int action)
{ 
	/*if(key == GLFW_KEY_F1)
    {
		MiroirManager::setShowMode(DEPTH_MAP);
	}
	else if(key == GLFW_KEY_F2)
	{
		MiroirManager::setShowMode(TSDF);
	}
	else if(key == GLFW_KEY_F3)
	{
		MiroirManager::setShowMode(RAYCAST);
	}*/

	if(key == 84 && action)	// t
    {
		MiroirManager::setRunTSDF(!MiroirManager::isTSDFRunning());
	}
}

void draw()
{
	glFinish();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT); // Clear required buffers  

	glBindVertexArray(vao);
	
	glDrawArrays(GL_TRIANGLES, 0, 6); // Starting from vertex 0; 3 vertices total -> 1 triangle
 
	glBindVertexArray(0);

	glfwSwapBuffers();
}

bool initGL()
{
	if(!glfwInit())
	{
		std::cerr << "Failed to initialize GLFW\n";
		return false;
	}
	
	glfwOpenWindowHint(GLFW_FSAA_SAMPLES, 4); // 4x antialiasing
	glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3); // We want OpenGL 3.3
	glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 3);
	glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //We don't want the old OpenGL
	glfwOpenWindowHint(GLFW_WINDOW_NO_RESIZE, GL_TRUE);
	// Open a window and create its OpenGL context
	if( !glfwOpenWindow( 640, 480, 0,0,0,0, 32,0, GLFW_WINDOW ) )
	{
		std::cerr <<  "Failed to open GLFW window\n";
		glfwTerminate();
		return false;
	}

	GLenum err = glGetError();
	assert(err == GL_NO_ERROR);

	glfwSetWindowTitle("MIROIR");
	glfwEnable( GLFW_STICKY_KEYS );
	glfwSetWindowSizeCallback( onWindowResized );
	glfwSetKeyCallback(keyboardCallback);
	err = glGetError();
	assert(err == GL_NO_ERROR);

	glewExperimental = GL_TRUE;
	if(glewInit() != GLEW_OK)
	{
		std::cerr <<  "Failed to open GLEW\n";
		return false;
	}
	glGetError();

	// background color
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	// fullscreen quad
	static const GLfloat g_quad_vertex_buffer_data[] = {
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		1.0f,  1.0f, 0.0f,
	};
 
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &quad_vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);
	glVertexAttribPointer(
	   0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
	   3,                  // size
	   GL_FLOAT,           // type
	   GL_FALSE,           // normalized?
	   0,                  // stride
	   (void*)0            // array buffer offset
	);

	glEnableVertexAttribArray(0); // Disable our Vertex Array Object  
	glBindVertexArray(0); // Disable our Vertex Buffer Object  

	// shaders
	GLuint vshader, fshader;
	GLuint program = GpuManager::createProgram("../src/Shaders/vert.glsl", "../src/Shaders/frag.glsl", vshader, fshader);

	glUseProgram(program);
	GLint texLoc = glGetUniformLocation(fshader, "texture");
	glUniform1i(texLoc, 0);	// TODO pass texture!

	int isValid;
	glValidateProgram(vshader);
	glGetProgramiv(vshader, GL_VALIDATE_STATUS, &isValid);
	assert(isValid);

	return true;
}

int main( int argc, char ** argv )
{
#ifdef MIROIR_GL_INTEROP
	if(!initGL())
		return -1;
#endif
	bool recompile_kernel = (argc == 2)&&(!strcmp(argv[1],"-kc"));
	
	GpuManager::initCL(recompile_kernel);
	
	MiroirManager& miroir = MiroirManager::getSingleton();

	miroir.initGlobalBuffers();
	miroir.buildMiroir();

#ifdef MIROIR_GL_INTEROP
	miroir.getGrabber()->start();
	do
	{
		draw();
		
		miroir.step();
		
	} while(glfwGetKey( GLFW_KEY_ESC ) != GLFW_PRESS && glfwGetWindowParam( GLFW_OPENED ));
	
#else
	boost::thread miroir_t( &MiroirManager::run, &miroir );
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("MIROIR"));
	viewer->setBackgroundColor (0, 0, 0);
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters();
	
	viewer->registerKeyboardCallback(&keyboardCallback);
	viewer->setCameraPosition(0.0, 0.0, -1.0, 0.0, -1.0, 0.0);

	miroir.setWindowDimensions(640, 480);

	//camera visualization
	//viewer->addCube(-0.05, 0.05, -0.05, 0.05, -0.05, 0.05);

	while(!viewer->wasStopped() || (glfwGetKey( GLFW_KEY_ESC ) != GLFW_PRESS && glfwGetWindowParam( GLFW_OPENED )))
	{
		pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud = miroir.getOutputCloud();
		if( cloud != NULL )
		{
			//viewer.showCloud(cloud);
			if(!viewer->updatePointCloud(cloud))
			{
				viewer->addPointCloud(cloud);
				viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1);
			}
			viewer->spinOnce (100);
		}

		cl_float4* camPose = miroir.getCameraPose();
		//viewer->addCube(camPose[3].s[0]*0.001-0.05, camPose[3].s[0]*0.001+0.05, camPose[3].s[1]*0.001-0.05, camPose[3].s[1]*0.001+0.05, camPose[3].s[2]*0.001-0.05, camPose[3].s[2]*0.001+0.05);
		
		pcl::ModelCoefficients coeffs;
		// translation
		coeffs.values.push_back (camPose[3].s[0]*0.001);
		coeffs.values.push_back (camPose[3].s[1]*0.001);
		coeffs.values.push_back (camPose[3].s[2]*0.001);
		// rotation
		float b1 = 0.5f * std::sqrt(1.0f + camPose[0].s[0] + camPose[1].s[1] + camPose[2].s[2]);
		coeffs.values.push_back (b1);
		coeffs.values.push_back ((camPose[2].s[1] - camPose[1].s[2]) / (4.0f*b1));
		coeffs.values.push_back ((camPose[0].s[2] - camPose[2].s[0]) / (4.0f*b1));
		coeffs.values.push_back ((camPose[1].s[0] - camPose[0].s[1]) / (4.0f*b1));
		// size
		coeffs.values.push_back (0.05);
		coeffs.values.push_back (0.05);
		coeffs.values.push_back (0.05);
		viewer->removeShape ("cube");
		viewer->addCube (coeffs, "cube");
		//boost::this_thread::sleep (boost::posix_time::microseconds (100000));
		//sleep(1);
	}


	MiroirManager::stop();
	miroir_t.join();
#endif 

	GpuManager::cleanCL();

	return 0;
}