cmake_minimum_required(VERSION 3.5)
project( mlp LANGUAGES CUDA CXX )

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 61)

add_executable( mlp unit_test_core.cu )

# Include CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries( mlp ${CUDA_LIBRARIES} )

# Include OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( mlp ${OpenCV_LIBS} )

# Include NCCL
include_directories(/usr/include) # NCCL include directory
target_link_libraries(mlp /usr/lib/libnccl.so) # Link against the NCCL shared library

# Include OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(mlp OpenMP::OpenMP_CXX)

