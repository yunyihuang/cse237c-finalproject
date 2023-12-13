############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
## Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
############################################################
open_project sepconv2d_optimized_bitwidth
set_top separable_conv_2d_cl
add_files cse237c-finalproject/sepconv2d_optimized_bitwidth/separable_conv_2d_cl.cpp
add_files cse237c-finalproject/sepconv2d_optimized_bitwidth/separable_conv_2d_cl.h
add_files -tb cse237c-finalproject/sepconv2d_optimized_bitwidth/separable_conv_2d_cl_test.cpp -cflags "-Wno-unknown-pragmas"
open_solution "solution1" -flow_target vivado
set_part {xc7z020-clg400-1}
create_clock -period 10 -name default
source "./sepconv2d_optimized_bitwidth/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
