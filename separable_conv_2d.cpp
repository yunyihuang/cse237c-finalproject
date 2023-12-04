#include <ap_fixed.h>
#include "separable_conv_2d.h"


// Depthwise convolution
// Flatten the 2d feature map into 1d array for easier operations
void depthwise_conv_2d_cl(
	ap_fixed<10,4> data[in_height * in_width * n_chan],
	ap_fixed<10,4> res[out_height * out_width * n_chan],
	ap_fixed<10,4> depthwise_weights[filt_height * filt_width * n_chan],
	ap_fixed<10,4> depthwise_biases[n_chan])
	{
		// depthwise output height
		for (int h = 0; h < in_height - filt_height + 1; h++) {
			// depthwise output width
			for (int w = 0; w < in_width - filt_width + 1; w++) {
				// number of channels
				for (int c = 0; c < n_chan; c++) {
					ap_fixed<10,4> sum = depthwise_biases[c];

					// kernel/filter multiplication
					for (int i = 0; i < filt_height; i++) {
						for (int j = 0; j < filt_width; j++) {

							int data_idx = (h+i) * in_width * n_chan + (w+j) * n_chan + c;
							int weight_idx = i * filt_width * n_chan + j * n_chan + c;
							std::cout << data[data_idx] << '\n';
							sum += data[data_idx] * depthwise_weights[weight_idx];
						}
					}
					int res_idx = (h * out_width * n_chan) + w * n_chan + c;
					res[res_idx] = sum;
				}
			}
		}
	}


// Pointwise convolution
// Flatten the 2d feature map into 1d array for easier operations
void pointwise_conv_2d_latency_cl(
	ap_fixed<10,4> data2[out_height * out_width * n_filt],
	ap_fixed<10,4> res2[out_height * out_width * n_filt],
	ap_fixed<10,4> pointwise_weights[n_chan * n_filt],
	ap_fixed<10,4> pointwise_biases[n_filt])
	{
		const int in_height2 = 4;
		const int in_width2 = 4;
		const int n_chan2 = 2;
		const int filt_height2 = 2;
		const int filt_width2 = 2;
		const int out_height2 = 2;
		const int out_width2 = 2;
		const int reuse_factor2 = 2;
		const int n_filt2 = 2;

		for (int h = 0; h < out_height2; h++) {
			for (int w = 0; w < out_width2; w++) {
				for (int f = 0; f < n_filt2; f++) {
					ap_fixed<10,4> sum = 0;
					//ap_fixed<10,4> sum = pointwise_biases[f];
					for (int c = 0; c < n_chan2; c++) {
						int data_idx = (h *out_width2 * n_chan2) + (w * n_chan2) + c;
						int weight_idx = c * n_filt2 + f;
						sum += data2[data_idx];// * pointwise_weights[weight_idx];
					}

					int res_idx = (h * out_width2 * n_filt2) + (w * n_filt2) + f;
					res2[res_idx] = sum;
				}
			}
		}
	}


void separable_conv_2d_cl(
	ap_fixed<10,4> data3[in_height * in_width * n_chan],
	ap_fixed<10,4> result[out_height * out_width * n_filt]){
	const int in_height2 = 10;
	const int in_width2 = 10;
	const int n_chan2 = 2;
	const int filt_height2 = 2;
	const int filt_width2 = 2;
	const int out_height2 = 2;
	const int out_width2 = 2;
	const int reuse_factor2 = 2;
	const int n_filt2 = 2;

	ap_fixed<10,4> res3[out_height2 * out_width2 * n_filt2] = {0,0,0,0,0,0,0,0};

	depthwise_conv_2d_cl(data3, res3); //depthwise_weights, depthwise_biases);
	pointwise_conv_2d_latency_cl(res3, result); //,pointwise_weights, pointwise_biases);
}
