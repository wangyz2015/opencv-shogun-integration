cmake_minimum_required(VERSION 2.8)

SET( SOURCE_FILES main.cpp facerecognizer_eigenfaces.cpp)


SET( CMAKE_CXX_FLAGS "" ) # compiler options


add_executable (eigenfaces ${SOURCE_FILES})

find_package(OpenCV REQUIRED)

TARGET_LINK_LIBRARIES(eigenfaces
	/usr/local/lib/libshogun.so
	${OpenCV_LIBS}
)
	

