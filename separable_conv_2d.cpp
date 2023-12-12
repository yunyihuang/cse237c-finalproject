#include <ap_fixed.h>
#include "separable_conv_2d.h"


// Depthwise convolution
// Flatten the 2d feature map into 1d array for easier operations
void depthwise_conv_2d_cl(
	int data[in_height * in_width * n_chan],
	int depthwise_res[out_height * out_width * n_chan],
	int depthwise_weights[filt_height * filt_width * n_chan],
	int depthwise_biases[n_chan])
	{
		for (int c = 0; c < n_chan; c++) {
			int channel_start = c * in_height * in_width;
			int kernel_start = c * filt_height * filt_width;

			for (int h = 0; h < out_height; h++) {
				for (int w = 0; w < out_width; w++) {
					int sum = depthwise_biases[c];
					for (int i = 0; i < filt_height; i++) {
						for (int j = 0; j < filt_width; j++) {
							int data_idx = channel_start + (h + i) * in_width + (w + j);
							int weight_idx = kernel_start + i * filt_width + j;
							sum += data[data_idx] * depthwise_weights[weight_idx];
						}
					}
					int res_idx = c * (out_height * out_width) + h * out_width + w;
					depthwise_res[res_idx] = sum;
 				}
			}
		}
	}


// Pointwise convolution
// Flatten the 2d feature map into 1d array for easier operations
void pointwise_conv_2d_latency_cl(
	int depthwise_res[out_height * out_width * n_chan],
	int res[out_height * out_width * n_filt],
	int pointwise_weights[n_chan * n_filt],
	int pointwise_biases[n_filt])
	{
		// pointwise output height
		for (int h = 0; h < out_height; h++) {
			// pointwise output width
			for (int w = 0; w < out_width; w++) {
				// output number of channels
				for (int f = 0; f < n_filt; f++) {
					int sum = pointwise_biases[f];

					// kernel/filter multiplication
					for (int c = 0; c < n_chan; c++) {
						int data_idx = c * (out_height * out_width) + h * out_width + w;
						int weight_idx = f * n_chan + c;
						sum += depthwise_res[data_idx] * pointwise_weights[weight_idx];
					}
					int res_idx = f * (out_height * out_width) + h * out_width + w;
					res[res_idx] = sum;
				}
			}
		}
	}


void separable_conv_2d_cl(
	int data[in_height * in_width * n_chan],
	int depthwise_res[out_height * out_width * n_chan],
	int res[out_height * out_width * n_filt],
	int depthwise_weights[filt_height * filt_width * n_chan],
	int pointwise_weights[n_chan * n_filt],
	int depthwise_biases[n_chan],
	int pointwise_biases[n_filt]){

	depthwise_conv_2d_cl(data, depthwise_res, depthwise_weights, depthwise_biases);
	pointwise_conv_2d_latency_cl(depthwise_res, res,pointwise_weights, pointwise_biases);
}
