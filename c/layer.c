#include "layer.h"

static function rectify(Number x)
{
        return x > 0.0 ? x : 0.0;
}

static function d_rectify(Number x)
{
        return x >= 0.0 ? 1.0 : 0.0;
}

struct Activation relu = {&rectify, &d_rectify};

struct Matrix prop_layer(const struct Layer *self, struct Matrix *input)
{
	for (int i = 0; i < input->m * input->n; i++)
		intput->a[i] = (*(self->activation.f))(a[i]);
	
	struct Matrix out = mat_mul(&self->w, &input
	mat_add_to(&out, &self->b);

	return out;
}
