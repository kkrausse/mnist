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

struct Activation relu = {&rectify, &d_rectify};

void prop_layer(Layer *self, Matrix *input);
void backprop_layer(Layer *self, Matrix *input);

#endif
