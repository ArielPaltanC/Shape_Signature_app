cmake_minimum_required(VERSION 3.10)
project(native_processor)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,max-page-size=16384")

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../../../../opencv-sdk/sdk/native/jni)
find_package(OpenCV REQUIRED)

add_library(native_processor SHARED native_processor.cpp)

target_include_directories(native_processor PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(native_processor
    ${OpenCV_LIBS}
    jnigraphics
    log
)
