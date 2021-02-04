#
# Copyright 2020 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

.PHONY: help

help::
	@echo  " Makefile Usage:"
	@echo  ""
	@echo  "  make build TARGET=sw_emu|hw_emu|hw "
	@echo  "      Command to build the application design "
	@echo  ""
	@echo  "  make run TARGET=sw_emu|hw_emu|hw "
	@echo  "      command to run the application design "
	@echo  ""
	@echo  "  make clean "
	@echo  "      Command to remove the generated files."
	@echo  ""

## platform selection
## Several supported platforms are liste here. You may choose the one you want and comment out the others
## Or you can add other platfom and set PLATFORM variable to it

PLATFORM := $(AWS_PLATFORM)
PLATFORM_NAME := xilinx_aws-vu9p-f1_shell-v04261818_201920_2


## TARGET can be set as:
## sw_emu: software emulation
## hw_emu: hardware Emulation
## hw: hardware run
TARGET := hw_emu

# Host Compiler Global Settings and Include Libraries

#CXXFLAGS :=
CXXFLAGS += -D__USE_XOPEN2K8
CXXFLAGS += -I$(XILINX_XRT)/include/
CXXFLAGS += -I$(XILINX_VIVADO)/include/
CXXFLAGS += -I$(SRC_REPO)
CXXFLAGS += -O2 -g -Wall -fmessage-length=0 -std=c++0x


CXXLDFLAGS := -L$(XILINX_XRT)/lib/
#CXXLDFLAGS += -lxilinxopencl -lpthread -lrt -lstdc++
CXXLDFLAGS += -lOpenCL -lpthread -lrt -lstdc++

## Kernel Compiler and Linker Flags

VPPFLAGS := -t $(TARGET)
VPPFLAGS += --platform $(PLATFORM)
VPPFLAGS += --config common.cfg



## Host Executable File Generation
build/:
	mkdir -p build/

build/main: src/main.cpp src/xcl2.cpp build/
	g++ $(CXXFLAGS) $(CXXLDFLAGS) -o $@ src/main.cpp src/xcl2.cpp

## Kernel XO and Xclbin File Generation
build/matmul_kernel.$(TARGET).$(PLATFORM_NAME).xo: src/matmul_kernel.cpp src/matmul_kernel.hpp build/
	v++ $(VPPFLAGS) -c src/matmul_kernel.cpp src/matmul_kernel.hpp -o $@
build/matmul_kernel.$(TARGET).$(PLATFORM_NAME).xclbin: build/matmul_kernel.$(TARGET).$(PLATFORM_NAME).xo build/
	v++ $(VPPFLAGS) -l -o $@ build/matmul_kernel.$(TARGET).$(PLATFORM_NAME).xo

## Emulation Files Generation
build/emconfig.json:
	 emconfigutil --nd 1  --platform $(PLATFORM) --od build/



.PHONY: all clean

## build the design without running host application
build: build/main build/matmul_kernel.$(TARGET).$(PLATFORM_NAME).xclbin build/emconfig.json


## build the design and then run host application
## run with 512x512 matrix size for hardware and 64x64 for emulation
run: build
	cp xrt.ini build;
ifeq ($(TARGET), hw)
	cd build && unset XCL_EMULATION_MODE; ./main;
else
	cd build && export XCL_EMULATION_MODE=$(TARGET); ./main;
endif


## Clean generated files

clean:
	rm -rf build
