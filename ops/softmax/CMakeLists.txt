cmake_minimum_required(VERSION 3.16)

# CPU 域孪生调试工程：不依赖昇腾硬件，用 g++ + tikicpulib 在 x86 CPU 上真实执行 Ascend C 核函数。
# 正常流程由 env 的 set_env.sh 提供 ASCEND_HOME_PATH；这里再加一层兜底，
# 让 VSCode / CMake Tools / 未 source 环境的终端也能直接 configure。
project(softmax CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # 给 cpptools / clangd 提供准确的编译参数

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    file(GLOB _cann_candidates "/usr/local/Ascend/cann-*")
    list(SORT _cann_candidates)
    if(_cann_candidates)
        list(GET _cann_candidates -1 _cann)
        set(ENV{ASCEND_HOME_PATH} "${_cann}")
        message(STATUS "ASCEND_HOME_PATH 未设置，兜底使用 ${_cann}")
    endif()
endif()
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "找不到 CANN：请先 source set_env.sh，或安装到 /usr/local/Ascend")
endif()

# 让 find_package(tikicpulib) 与 ascendcl 都能在不 source 环境时解析
list(APPEND CMAKE_PREFIX_PATH
    "$ENV{ASCEND_HOME_PATH}/tools/tikicpulib/lib/cmake"
    "$ENV{ASCEND_HOME_PATH}/x86_64-linux/lib64/cmake"
)
link_directories("$ENV{ASCEND_HOME_PATH}/x86_64-linux/lib64")

if(NOT DEFINED soc_version)
    set(soc_version "Ascend910B3")
endif()

find_package(tikicpulib REQUIRED)

add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/softmax.cpp)

target_compile_options(${PROJECT_NAME} PRIVATE
    -g -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ascendcl
    tikicpulib::${soc_version}
)
