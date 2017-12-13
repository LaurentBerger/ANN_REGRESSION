cmake_minimum_required(VERSION 2.8)

PROJECT (ANN_Regression)

find_package(OpenCV )


file(GLOB ANN_Regression_SRCS
    "*.h"
    "*.cpp")
ADD_EXECUTABLE (ANN_Regression ${ANN_Regression_SRCS})



if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	target_link_libraries( ANN_Regression ${OpenCV_LIBS} )
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)

