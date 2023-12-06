#include "separable_conv_2d.h"
#include <iostream>


int main(){
	ap_fixed<10,4> data[in_height * in_width * n_chan] = {1,2,3,4,5,6,7,8,9};
	ap_fixed<10,4> res[out_height * out_width * n_chan] = {0,0,0,0,0,0};
	ap_fixed<10,4> kernel[filt_height * filt_width * n_chan] = {3,7};
	ap_fixed<10,4> biases[n_chan] = {0};

	depthwise_conv_2d_cl(data, res, kernel, biases);

	for (int i = 0; i < 6; i++){
		std::cout << res[i] << '\n';
	}

	return 0;
};
