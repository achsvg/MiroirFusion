cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(MIRROIR)

find_package(PCL 1.3 REQUIRED COMPONENTS common io visualization)
find_package(Boost COMPONENTS timer REQUIRED chrono REQUIRED)

# TODO : figure out good compiler options
if(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

file(GLOB incs include/*.h)
file(GLOB srcs src/*.cpp)
file(GLOB krnls src/OpenCL/*.cl)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# path to cl kernel files
set( KERNEL_PATH "\"${PROJECT_SOURCE_DIR}/src/OpenCL/\"" )
configure_file ( "${PROJECT_SOURCE_DIR}/include/config.h.in" "${PROJECT_BINARY_DIR}/config.h" )
include_directories("${PROJECT_BINARY_DIR}")

#OpenCL
find_package(OPENCL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS} ${GLFW_INCLUDE_PATH} ${GLEW_INCLUDE_PATH})

if( OPENCL_HAS_CPP_BINDINGS )
	message( "OpenCL has CPP bindings. Full include is: " ${OPENCL_INCLUDE_DIRS} )
else( OPENCL_HAS_CPP_BINDINGS )
	message( "No OpenCL CPP bindings found" )
endif( OPENCL_HAS_CPP_BINDINGS )

add_executable(miroir ${srcs} ${incs} ${krnls})
target_link_libraries(miroir ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${OPENCL_LIBRARIES} ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${Boost_TIMER_LIBRARY} ${Boost_CHRONO_LIBRARY})
#add_subdirectory(src)