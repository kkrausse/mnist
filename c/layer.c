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
	
/*
 * propagates the input through the layer and returns the output matrix.
 */
struct Matrix prop_layer(const struct Layer *self, struct Matrix *input)
{
	for (int i = 0; i < input->m * input->n; i++)
		intput->a[i] = (*(self->activation.f))(a[i]);
	
	struct Matrix out = mat_mul(&self->w, &input);
	mat_add_to(&out, &self->b);

	return out;
}

/*
 * gradient_buf is the current sum of gradients with respect to the layer.
 * TODO: implement this with a lock on gradient_buf later for other threads.
 *
 * h is the value that was originally fed into this layer fo rthe output.
 * gradient is the gradient going into the layer and returns the gradient going out.
 */
struct Matrix backprop_layer(const struct Layer *self,
				struct Matrix gradient,
				struct Matrix *h,
				struct Layer *gradient_buf)
{
	//recompute f(h) to save some space
	struct Matrix fh = new_mat(h->m, h->n);

	for (int i = 0; i < fh.m * fh.n; i++)
		fh.a[i] = (*(self->activation.f))(h->a[i]);

	struct Matrix w_grad = mat_mul_tr(&gradient, &fh);

	mat_add_to(&gradient_buf->b, &gradient);
	mat_add_to(&gradient_buf->w, &w_grad);
	
	
}
