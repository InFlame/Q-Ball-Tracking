#define CURV = 0;

__kernel void write(__global float* output) {
	output[get_global_id(0)] = -1.0f;
}