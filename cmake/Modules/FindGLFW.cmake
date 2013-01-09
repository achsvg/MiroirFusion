#
# Try to find GLFW library and include path.
# Once done this will define
#
# GLFW_FOUND
# GLFW_INCLUDE_PATH
# GLFW_LIBRARY
# 

IF (WIN32)
	FIND_PATH( GLFW_INCLUDE_PATH GL/glfw.h
		$ENV{PROGRAMFILES}/GLFW/include
		${PROJECT_SOURCE_DIR}/src/nvgl/glfw/include
		DOC "The directory where GL/glfw.h resides")
	FIND_LIBRARY( GLFW_LIBRARY
		NAMES glfw GLFW glfw32 glfw32s
		PATHS
		$ENV{PROGRAMFILES}/GLFW/lib
		${PROJECT_SOURCE_DIR}/src/nvgl/glfw/bin
		${PROJECT_SOURCE_DIR}/src/nvgl/glfw/lib
		DOC "The GLFW library")
ELSE (WIN32)
	FIND_PATH( GLFW_INCLUDE_PATH GL/glfw.h
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
		DOC "The directory where GL/glfw.h resides")
	FIND_LIBRARY( GLFW_LIBRARY
		NAMES GLFW glfw
		PATHS
		/usr/lib64
		/usr/lib
		/usr/local/lib64
		/usr/local/lib
		/sw/lib
		/opt/local/lib
		DOC "The GLFW library")
ENDIF (WIN32)

IF (GLFW_INCLUDE_PATH)
	SET( GLFW_FOUND 1 CACHE STRING "Set to 1 if GLFW is found, 0 otherwise")
ELSE (GLFW_INCLUDE_PATH)
	SET( GLFW_FOUND 0 CACHE STRING "Set to 1 if GLFW is found, 0 otherwise")
ENDIF (GLFW_INCLUDE_PATH)

MARK_AS_ADVANCED( GLFW_FOUND )