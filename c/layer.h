#ifndef LAYER
#define LAYER

#include "matrix.h"

struct Activation {
	Number (*f)(Number);
	Number (*df)(Number);
}

struct Layer {
	int in;
	int out;

	Activation activation;
	Matrix w;
	Matrix b;
}

struct Matrix prop_layer(struct Layer *self, struct Matrix *input);
struct Matrix backprop_layer(const struct Layer *self,
				struct Matrix gradient,
				struct Layer *gradient_buf);

#endif
