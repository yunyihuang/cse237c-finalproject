#include "separable_conv_2d_cl.h"


// Depthwise convolution
// Flatten the 2d feature map into 1d array for easier operations
void depthwise_conv_2d_cl(
	rgb_data_t data[in_height * in_width * n_chan],
	conv_data_t depthwise_res[out_height * out_width * n_chan],
	weight_t depthwise_weights[filt_height * filt_width * n_chan],
	conv_data_t depthwise_biases[n_chan])
	{
#pragma HLS array_partition variable=data complete dim=1
#pragma HLS array_partition variable=depthwise_weights complete dim=1
#pragma HLS array_partition variable=depthwise_res complete dim=1
		// multiple channels of input
		for (int c = 0; c < n_chan; c++) {
			int channel_start = c * in_height * in_width;
			int kernel_start = c * filt_height * filt_width;
			// depthwise output height
			for (int h = 0; h < out_height; h++) {
				// depthwise output width
				for (int w = 0; w < out_width; w++) {
					conv_data_t sum = depthwise_biases[c];

					// kernel multiplication
					for (int i = 0; i < filt_height; i++) {
						for (int j = 0; j < filt_width; j++) {
							int data_idx = channel_start + (h + i) * in_width + (w + j);
							int weight_idx = kernel_start + i * filt_width + j;
							sum += static_cast<conv_data_t>(data[data_idx]) * depthwise_weights[weight_idx];
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
	conv_data_t depthwise_res[out_height * out_width * n_chan],
	conv_data_t res[out_height * out_width * n_filt],
	weight_t pointwise_weights[n_chan * n_filt],
	conv_data_t pointwise_biases[n_filt])
	{
#pragma HLS array_partition variable=depthwise_res complete dim=1
#pragma HLS array_partition variable=pointwise_weights complete dim=2
#pragma HLS array_partition variable=res complete dim=1
		// pointwise output height
		for (int h = 0; h < out_height; h++) {
			// pointwise output width
			for (int w = 0; w < out_width; w++) {
				// output number of channels
				for (int f = 0; f < n_filt; f++) {
					conv_data_t sum = pointwise_biases[f];

					// kernel multiplication
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
	rgb_data_t data[in_height * in_width * n_chan],
	conv_data_t depthwise_res[out_height * out_width * n_chan],
	conv_data_t res[out_height * out_width * n_filt],
	weight_t depthwise_weights[filt_height * filt_width * n_chan],
	weight_t pointwise_weights[n_chan * n_filt],
	conv_data_t depthwise_biases[n_chan],
	conv_data_t pointwise_biases[n_filt]){

#pragma HLS dataflow

	depthwise_conv_2d_cl(data, depthwise_res, depthwise_weights, depthwise_biases);
	pointwise_conv_2d_latency_cl(depthwise_res, res,pointwise_weights, pointwise_biases);
}
