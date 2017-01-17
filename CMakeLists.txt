cmake_minimum_required(VERSION 3.6)
project(ppd_projects)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )

find_package( CUDA REQUIRED )

find_package( MPI REQUIRED )

add_executable(ppd_projects main.cpp)
target_link_libraries(ppd_projects -lpthread ${OpenCV_LIBS})

cuda_add_executable(ppd_projects_cuda main_cuda.cpp cuda.h cuda.cu)
target_link_libraries(ppd_projects_cuda ${OpenCV_LIBS})