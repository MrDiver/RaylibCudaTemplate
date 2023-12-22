cmake_minimum_required(VERSION 3.20)
project(octree)

enable_language(CUDA) # Enables CUDA
find_package(CUDAToolkit REQUIRED) # Finds CUDA Toolkit and sets CUDA_TOOLKIT_ROOT_DIR
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_CXX_STANDARD 17) # Requires C++17 standard

add_subdirectory(raylib)
add_executable(${PROJECT_NAME} src/main.cpp src/kernel.cu)
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_link_libraries(${PROJECT_NAME} raylib)

# Checks if OSX and links appropriate frameworks (only required on MacOS)
if (APPLE)
    target_link_libraries(${PROJECT_NAME} "-framework IOKit")
    target_link_libraries(${PROJECT_NAME} "-framework Cocoa")
    target_link_libraries(${PROJECT_NAME} "-framework OpenGL")
endif()
