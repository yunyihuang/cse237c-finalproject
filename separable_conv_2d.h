#include <ap_fixed.h>

// Input matrix's height and weight
const int in_height =3;
const int in_width = 3;
// Number of channels
const int n_chan = 3;
// Depthwise kernel's height and weight
const int filt_height = 2;
const int filt_width = 2;
// Intermediate matrix's height and weight
const int out_height = in_height - filt_height + 1;
const int out_width = in_width - filt_width + 1;
// Number of output channels
const int n_filt = 2;

void depthwise_conv_2d_cl(
	int data[in_height * in_width * n_chan],
	int depthwise_res[out_height * out_width * n_chan],
	int depthwise_weights[filt_height * filt_width * n_chan],
	int depthwise_biases[n_chan]
	);


void pointwise_conv_2d_latency_cl(
	int depthwise_res[out_height * out_width * n_chan],
	int res[out_height * out_width * n_filt],
	int pointwise_weights[n_chan * n_filt],
	int pointwise_biases[n_filt]
	);


void separable_conv_2d_cl(
	int data[in_height * in_width * n_chan],
	int depthwise_res[out_height * out_width * n_chan],
	int res[out_height * out_width * n_filt],
	int depthwise_weights[filt_height * filt_width * n_chan],
	int pointwise_weights[n_chan * n_filt],
	int depthwise_biases[n_chan],
	int pointwise_biases[n_filt]
	);
