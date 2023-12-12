#include "separable_conv_2d_cl.h"


// Depthwise convolution
// Flatten the 2d feature map into 1d array for easier operations
void depthwise_conv_2d_cl(
	rgb_data_t data[in_height * in_width * n_chan],
	conv_data_t depthwise_res[out_height * out_width * n_chan],
	weight_t depthwise_weights[filt_height * filt_width * n_chan],
	conv_data_t depthwise_biases[n_chan])
	{
		// multiple channels of input
		for (int c = 0; c < n_chan; c++) {
#pragma HLS pipeline II=2
			int channel_start = c * in_height * in_width;
			int kernel_start = c * filt_height * filt_width;
			// depthwise output height
			for (int h = 0; h < out_height; h++) {
#pragma HLS pipeline II=2
				// depthwise output width
				for (int w = 0; w < out_width; w++) {
					conv_data_t sum = depthwise_biases[c];
#pragma HLS pipeline II=2
					// kernel multiplication
					for (int i = 0; i < filt_height; i++) {
#pragma HLS pipeline II=2
						for (int j = 0; j < filt_width; j++) {
#pragma HLS pipeline II=2
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
		// pointwise output height
		for (int h = 0; h < out_height; h++) {
#pragma HLS pipeline II=2
			// pointwise output width
			for (int w = 0; w < out_width; w++) {
#pragma HLS pipeline II=2
				// output number of channels
				for (int f = 0; f < n_filt; f++) {
#pragma HLS pipeline II=2
					conv_data_t sum = pointwise_biases[f];

					// kernel multiplication
					for (int c = 0; c < n_chan; c++) {
#pragma HLS pipeline II=2
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

	depthwise_conv_2d_cl(data, depthwise_res, depthwise_weights, depthwise_biases);
	pointwise_conv_2d_latency_cl(depthwise_res, res,pointwise_weights, pointwise_biases);
}
