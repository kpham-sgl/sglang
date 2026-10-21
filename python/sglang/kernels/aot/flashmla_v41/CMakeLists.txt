# Standalone build of flashmla_v41_ops (see ../cmake/flashmla_v41.cmake) against the torch of the
# running Python, without configuring the rest of sgl-kernel. Driven by build.sh.
cmake_minimum_required(VERSION 3.26)
project(flashmla_v41 LANGUAGES CXX CUDA)
include(FetchContent)
set(GITHUB_ARTIFACTORY "github.com" CACHE STRING "GitHub mirror URL")
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS "13.1")
    message(FATAL_ERROR "flashmla_v41_ops needs nvcc >= 13.1, got ${CUDAToolkit_VERSION}")
endif()
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch.utils.cpp_extension as e; print(';'.join(e.include_paths()))"
    OUTPUT_VARIABLE FLASHMLA_V41_TORCH_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch.utils.cpp_extension as e; print(e.library_paths()[0])"
    OUTPUT_VARIABLE _TORCH_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
set(FLASHMLA_V41_TORCH_LIBRARIES
    ${_TORCH_LIB_DIR}/libc10.so ${_TORCH_LIB_DIR}/libc10_cuda.so
    ${_TORCH_LIB_DIR}/libtorch.so ${_TORCH_LIB_DIR}/libtorch_cpu.so ${_TORCH_LIB_DIR}/libtorch_cuda.so)
set(FLASHMLA_V41_TORCH_PYTHON_LIBRARY ${_TORCH_LIB_DIR}/libtorch_python.so)
set(FLASHMLA_V41_PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE _CXX11_ABI OUTPUT_STRIP_TRAILING_WHITESPACE)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${_CXX11_ABI})
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/flashmla_v41.cmake)
