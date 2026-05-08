cmake_minimum_required(VERSION 3.16)
project(gesture_detector CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# ─── OpenCV ──────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# ─── ONNX Runtime ────────────────────────────────────────────────────────────
# Usage:
#   cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.25.1
#
# ORT 1.20+ puts headers in:
#   include/onnxruntime_cxx_api.h          (older layout)
#   include/onnxruntime/core/session/...   (newer layout, header-only at root)
# We search both and add ALL candidate dirs to be safe.

if(NOT DEFINED ONNXRUNTIME_ROOT)
    message(FATAL_ERROR
        "Please specify ONNXRUNTIME_ROOT:\n"
        "  cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.25.1")
endif()

# Normalise path (removes leading // or trailing /)
get_filename_component(ORT_ROOT "${ONNXRUNTIME_ROOT}" REALPATH)
message(STATUS "ORT root (resolved): ${ORT_ROOT}")

# Collect every include sub-directory that actually exists
set(ORT_INCLUDE_DIRS "")
foreach(_d
    "${ORT_ROOT}/include"
    "${ORT_ROOT}/include/onnxruntime"
    "${ORT_ROOT}/include/onnxruntime/core/session"
)
    if(EXISTS "${_d}")
        list(APPEND ORT_INCLUDE_DIRS "${_d}")
        message(STATUS "ORT include dir: ${_d}")
    endif()
endforeach()

if(NOT ORT_INCLUDE_DIRS)
    message(FATAL_ERROR "No include directory found under ${ORT_ROOT}/include")
endif()

# Locate the shared library (handles .so, .so.1.x, symlinks)
find_library(ORT_LIB
    NAMES onnxruntime
    PATHS "${ORT_ROOT}/lib"
    NO_DEFAULT_PATH
)
if(NOT ORT_LIB)
    message(FATAL_ERROR "libonnxruntime not found in ${ORT_ROOT}/lib")
endif()
message(STATUS "ORT lib: ${ORT_LIB}")

# ─── Target ──────────────────────────────────────────────────────────────────
add_executable(gesture_detector gesture_detector.cpp)

target_include_directories(gesture_detector PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ORT_INCLUDE_DIRS}
)

target_link_libraries(gesture_detector
    ${OpenCV_LIBS}
    "${ORT_LIB}"
)

# ─── RPATH so the binary finds libonnxruntime.so at runtime ──────────────────
if(UNIX)
    get_filename_component(ORT_LIB_DIR "${ORT_LIB}" DIRECTORY)
    set_target_properties(gesture_detector PROPERTIES
        BUILD_RPATH   "${ORT_LIB_DIR}"
        INSTALL_RPATH "${ORT_LIB_DIR}"
    )
endif()
