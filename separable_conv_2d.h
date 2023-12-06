#include <ap_fixed.h>

// Input matrix's height and weight
const int in_height =3;
const int in_width = 3;
// Number of channels
const int n_chan = 1;
// Depthwise kernel's height and weight
const int filt_height = 2;
const int filt_width = 1;
// Intermediate matrix's height and weight
const int out_height = in_height - filt_height + 1;
const int out_width = in_width - filt_width + 1;
// Number of output channels
const int n_filt = 1;

void depthwise_conv_2d_cl(
	ap_fixed<10,4> data[in_height * in_width * n_chan],
	ap_fixed<10,4> depthwise_res[out_height * out_width * n_chan],
	ap_fixed<10,4> depthwise_weights[filt_height * filt_width * n_chan],
	ap_fixed<10,4> depthwise_biases[n_chan]
	);


void pointwise_conv_2d_latency_cl(
	ap_fixed<10,4> depthwise_res[out_height * out_width * n_chan],
	ap_fixed<10,4> res[out_height * out_width * n_filt],
	ap_fixed<10,4> pointwise_weights[n_chan * n_filt], // shouldn't it always be 1 x 1?
	ap_fixed<10,4> pointwise_biases[n_filt]
	);


void separable_conv_2d_cl(
	ap_fixed<10,4> data[in_height * in_width * n_chan],
	ap_fixed<10,4> depthwise_res[out_height * out_width * n_chan],
	ap_fixed<10,4> res[out_height * out_width * n_filt],
	ap_fixed<10,4> depthwise_weights[filt_height * filt_width * n_chan],
	ap_fixed<10,4> pointwise_weights[n_chan * n_filt],
	ap_fixed<10,4> depthwise_biases[n_chan],
	ap_fixed<10,4> pointwise_biases[n_filt]
	);
