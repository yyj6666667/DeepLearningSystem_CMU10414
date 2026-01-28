cmake_minimum_required(VERSION 3.10)
project(needle C CXX)

# find correct version of Python
find_package(Python COMPONENTS Development Interpreter REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# find pybind
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT __pybind_exit_code EQUAL 0)
  message(FATAL_ERROR "pybind11 not found. Please install it with 'pip install pybind11'")
endif()

find_package(pybind11 PATHS ${__pybind_path} REQUIRED)

if(MSVC)
  set(CMAKE_CXX_FLAGS "/std:c++11 /O2 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
endif()

###################
### CPU BACKEND ###
###################
pybind11_add_module(ndarray_backend_cpu src/ndarray_backend_cpu.cc)

# directly output to ffi folder
set_target_properties(ndarray_backend_cpu
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray"
  PREFIX ""
)

####################
### CUDA BACKEND ###
####################
# Optional CUDA support - disabled by default on Windows MinGW as it's tricky
option(ENABLE_CUDA "Enable CUDA backend" OFF)
if(ENABLE_CUDA)
  find_package(CUDA)
  if(CUDA_FOUND)
    message(STATUS "Found cuda, building cuda backend")
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    
    # set arch flags properly
    cuda_add_library(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu)
    target_link_libraries(ndarray_backend_cuda PRIVATE pybind11::module)
    
    set_target_properties(ndarray_backend_cuda
      PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray"
      PREFIX ""
    )
  endif()
endif()
