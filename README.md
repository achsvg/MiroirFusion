MiroirFusion
============

This is an OpenCL version of the KinectFusion (http://research.microsoft.com/en-us/projects/surfacerecon/).
The OpenCL code is mainly inspired by the CUDA implementation from the PCL project (http://pointclouds.org/).

As I don't have a lot of time to dedicate to this project anymore, I just decided to release it to the public and maybe you guys will contribute to improving it :)

The code is far from being optimal nor clean, however from my testings it is working (I'm using a GT330) at 5-10 fps. 

The application uses the PCL visualizer and can also use GLFW to show the raycast from the camera (uncomment GL_INTEROP in gpu_def.h).

Compiling
---------
CMake included.
Requires PCL, GLFW, Boost, GLEW. First install PCL and all its dependencies (Boost, Qt) then unzip GLFW and GLEW. Use CMake to point to GLFW and GLEW's library and include folder. Generate the solution files with CMake.

Commands (for PCL visualizer only)
-----------------------------------
F1 : show the raw depth map from depth sensor  
F2 : show the result of bilateral filtering on raw depth map  
F3 : show the TSDF   
F4 : show the raycasted surface normal map  
F5 : show the corresponding vertices from filtered depth map and raycasted surface  

