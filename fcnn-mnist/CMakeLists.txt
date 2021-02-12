cmake_minimum_required(VERSION 3.17)
project(fcnn-mnist)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../third_party/hlslib/cmake")
message(STATUS "CMAKE_MODULE_PATH" ${CMAKE_MODULE_PATH})
find_package(Vitis REQUIRED)

set(CMAKE_CXX_STANDARD 11)

add_executable(main src/main.cpp src/xcl2.cpp)
target_include_directories(
    main PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../third_party/optional-lite/include"
)
include_directories(${Vitis_INCLUDE_DIRS})
target_link_libraries(main ${Vitis_LIBRARIES})

if(NOT TARGET)
    set(TARGET sw_emu)
endif()
if(NOT HW_PLATFORM)
   set(HW_PLATFORM $ENV{AWS_PLATFORM})
endif()

message(STATUS "HW_PLATFORM" ${HW_PLATFORM})
add_custom_target(compile_kernel ${Vitis_COMPILER}
                  -c -t ${TARGET} ${CMAKE_CURRENT_LIST_DIR}/src/matmul_kernel.cpp
                  --kernel matmul_kernel
                  --platform ${HW_PLATFORM}
                  -o xclbin/matmul_kernel.xo
                  BYPRODUCTS xclbin/matmul_kernel.xo
                  )
add_custom_target(kernel
                  CPATH=${CMAKE_CURRENT_LIST_DIR}/src
                  ${Vitis_COMPILER}
                  -l -t ${TARGET} xclbin/matmul_kernel.xo
                  --kernel matmul_kernel
                  --platform ${HW_PLATFORM}
                  -o xclbin/matmul_kernel.xclbin
                  DEPENDS compile_kernel
                  BYPRODUCTS xclbin/matmul_kernel.xclbin)

add_custom_target(emconfig.json emconfigutil --nd 1  --platform ${HW_PLATFORM})
# add_dependencies(main kernel)