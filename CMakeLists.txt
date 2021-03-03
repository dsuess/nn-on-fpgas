cmake_minimum_required(VERSION 3.17)
project(fcnn-mnist)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/third_party/hlslib/cmake")
message(STATUS "CMAKE_MODULE_PATH" ${CMAKE_MODULE_PATH})
find_package(Vitis REQUIRED)

set(CMAKE_CXX_STANDARD 11)

add_executable(main src/main.cpp src/xcl2.cpp)
target_include_directories(
    main PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/third_party/optional-lite/include"
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
add_custom_target(kernels
                  CPATH=${CMAKE_CURRENT_LIST_DIR}/src
                  ${Vitis_COMPILER}
                  -l -t ${TARGET} xclbin/*.xo
                  --platform ${HW_PLATFORM}
                  -o xclbin/kernels.xclbin
                  BYPRODUCTS xclbin/kernels.xclbin)
function(compile_kernel kernel_name)
add_custom_target(compile_${kernel_name} ${Vitis_COMPILER}
                  -c -t ${TARGET} ${CMAKE_CURRENT_LIST_DIR}/src/${kernel_name}.cpp
                  --kernel ${kernel_name}
                  --platform ${HW_PLATFORM}
                  -o xclbin/${kernel_name}.xo
                  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/src/${kernel_name}.cpp
                  BYPRODUCTS xclbin/${kernel_name}.xo
                  )
add_dependencies(kernels compile_${kernel_name})
endfunction()

compile_kernel(matmul_kernel)
compile_kernel(bias_relu6_kernel)
compile_kernel(bias_softmax_kernel)

add_custom_target(emconfig.json emconfigutil --nd 1  --platform ${HW_PLATFORM})
# add_dependencies(main kernel)
