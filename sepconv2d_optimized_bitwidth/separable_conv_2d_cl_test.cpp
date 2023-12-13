#include "separable_conv_2d_cl.h"
#include <iostream>


int main(){
	rgb_data_t data[in_height * in_width * n_chan] = {1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,1,2,4,6,8,10,12,14,16,18};
	conv_data_t dw_res[out_height * out_width * n_chan] = {0,0,0,0,0,0,0,0,0,0,0,0};
	conv_data_t res[out_height * out_width * n_filt] = {0,0,0,0,0,0,0,0};
	weight_t dw_kernel[filt_height * filt_width * n_chan] = {1,0,0,1,1,0,0,1,1,0,0,1};
	weight_t pw_kernel[n_chan * n_filt] = {1,0,1,0,1,1};
	conv_data_t dw_biases[n_chan] = {0,0,0};
	conv_data_t pw_biases[n_filt] = {0,0};

	separable_conv_2d_cl(data, dw_res, res, dw_kernel, pw_kernel, dw_biases, pw_biases);

	for (int i = 0; i < 8; i++){
		std::cout << res[i] << '\n';
	}

	return 0;
};
