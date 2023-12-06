#include <ap_fixed.h>
#include "separable_conv_2d.h"


// Depthwise convolution
// Flatten the 2d feature map into 1d array for easier operations
void depthwise_conv_2d_cl(
	ap_fixed<10,4> data[in_height * in_width * n_chan],
	ap_fixed<10,4> depthwise_res[out_height * out_width * n_chan],
	ap_fixed<10,4> depthwise_weights[filt_height * filt_width * n_chan],
	ap_fixed<10,4> depthwise_biases[n_chan])
	{
		// depthwise output height
		for (int h = 0; h < out_height; h++) {
			// depthwise output width
			for (int w = 0; w < out_width + 1; w++) {
				// input number of channels
				for (int c = 0; c < n_chan; c++) {
					ap_fixed<10,4> sum = depthwise_biases[c];

					// kernel/filter multiplication
					for (int i = 0; i < filt_height; i++) {
						for (int j = 0; j < filt_width; j++) {

							int data_idx = (h+i) * in_width * n_chan + (w+j) * n_chan + c;
							int weight_idx = i * filt_width * n_chan + j * n_chan + c;
							sum += data[data_idx] * depthwise_weights[weight_idx];
						}
					}
					int res_idx = (h * out_width * n_chan) + w * n_chan + c;
					depthwise_res[res_idx] = sum;
				}
			}
		}
	}


// Pointwise convolution
// Flatten the 2d feature map into 1d array for easier operations
void pointwise_conv_2d_latency_cl(
	ap_fixed<10,4> depthwise_res[out_height * out_width * n_chan],
	ap_fixed<10,4> res[out_height * out_width * n_filt],
	ap_fixed<10,4> pointwise_weights[n_chan * n_filt],
	ap_fixed<10,4> pointwise_biases[n_filt])
	{
		// pointwise output height
		for (int h = 0; h < out_height; h++) {
			// pointwise output width
			for (int w = 0; w < out_width; w++) {
				// output number of channels
				for (int f = 0; f < n_filt; f++) {
					ap_fixed<10,4> sum = pointwise_biases[f];

					// kernel/filter multiplication
					for (int c = 0; c < n_chan; c++) {
						int data_idx = (h *out_width * n_chan) + (w * n_chan) + c;
						int weight_idx = c * n_filt + f;
						sum += depthwise_res[data_idx] * pointwise_weights[weight_idx];
					}
					int res_idx = (h * out_width * n_filt) + (w * n_filt) + f;
					res[res_idx] = sum;
				}
			}
		}
	}


void separable_conv_2d_cl(
	ap_fixed<10,4> data[in_height * in_width * n_chan],
	ap_fixed<10,4> depthwise_res[out_height * out_width * n_chan],
	ap_fixed<10,4> res[out_height * out_width * n_filt],
	ap_fixed<10,4> depthwise_weights[filt_height * filt_width * n_chan],
	ap_fixed<10,4> pointwise_weights[n_chan * n_filt],
	ap_fixed<10,4> depthwise_biases[n_chan],
	ap_fixed<10,4> pointwise_biases[n_filt]){

	depthwise_conv_2d_cl(data, depthwise_res, depthwise_weights, depthwise_biases);
	pointwise_conv_2d_latency_cl(depthwise_res, res,pointwise_weights, pointwise_biases);
}
