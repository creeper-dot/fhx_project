cmake_minimum_required(VERSION 3.16)
project(Ballistic_Calculation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

find_path(SERIAL_INCLUDE_DIR NAMES serial/serial.h PATHS /usr/local/include REQUIRED)
find_library(SERIAL_LIBRARY NAMES serial PATHS /usr/local/lib REQUIRED)

set(MVS_INCLUDE_DIR "/opt/MVS/include")
set(MVS_LIBRARY "/opt/MVS/lib/64/libMvCameraControl.so")

if(NOT EXISTS ${MVS_INCLUDE_DIR}/MvCameraControl.h)
    message(FATAL_ERROR "MvCameraControl.h not found at ${MVS_INCLUDE_DIR}")
endif()

if(NOT EXISTS ${MVS_LIBRARY})
    message(FATAL_ERROR "MvCameraControl library not found at ${MVS_LIBRARY}")
endif()

message(STATUS "MVS Include: ${MVS_INCLUDE_DIR}")
message(STATUS "MVS Library: ${MVS_LIBRARY}")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/io
    ${CMAKE_CURRENT_SOURCE_DIR}/tasks
    ${CMAKE_CURRENT_SOURCE_DIR}/tools
    ${MVS_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${SERIAL_INCLUDE_DIR}
)

add_executable(${PROJECT_NAME}
    src/main.cpp
    io/camera.cpp
    tasks/detector.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    fmt::fmt
    ${SERIAL_LIBRARY}
    ${MVS_LIBRARY}
    pthread
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)