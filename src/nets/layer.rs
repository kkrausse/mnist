use nets::*;
use std::sync::Mutex;

pub struct Layer<A: Activation<Number>>
{
    pub w: Matrix<Number>,
    b: Matrix<Number>,
    activation: A,
}

impl<A: Activation<Number>> Layer<A>
{
    pub fn new_rand(activation: A, into: usize, out: usize)
                    -> Layer<A>
    {
	let w_max = 1.0 /((into * out) as Number).sqrt();
        Layer { w: Matrix::new_rand((out, into), -1.0 * w_max, w_max),
                b: Matrix::new_const((out, 1), 0.01),
                activation }
    }

    pub fn clone_zeros(&self) -> Layer<A>
    {
        Layer {
            w: Matrix::new_const(self.w.dim, 0.0),
            b: Matrix::new_const(self.b.dim, 0.0),
            activation: self.activation.clone(),
        }
    }

    pub fn update(&mut self, other: &Layer<A>, batch_size: usize, step: Number)
    {
        for (i, e) in &mut self.w.a.iter_mut().enumerate() {
            *e = *e - (other.w.a[i] / batch_size as Number) * step;
        }

        for (i, e) in self.b.a.iter_mut().enumerate() {
            *e = *e - (other.b.a[i] / batch_size as Number) * step;
        }
    }

    pub fn zero_out(&mut self)
    {
        for e in &mut self.w.a {
            *e = 0.0;
        }

        for e in &mut self.b.a {
            *e = 0.0;
        }
    }

    pub fn prop(&self, mut x: Matrix<Number>) -> Matrix<Number>
    {
        for e in x.a.iter_mut() {
		*e = self.activation.f(*e);
	}
        let mut out = &self.w * &x;
        out.add_by(&self.b);
        out
    }

    /*
     * TODO: Add regularization terms for layer parameters.
     *	And do a so i can update the gradient_buf in multiple threads.
     */
    pub fn backprop(&self,
                    gradient: Matrix<Number>,
                    mut x: Matrix<Number>,
                    gradient_buf: &Mutex<Layer<A>>)
                    -> Matrix<Number>
    {
	let alpha = 0.09;


        let mut out_gradient = self.w.mul_tl(&gradient); //same as (w^t)*g

        //hadamar product of g with df with respect to the input.
        for i in 0..out_gradient.len() {
            out_gradient.a[i] = out_gradient[i] * self.activation.df(x[i]);
        }

        //recompute f(x)
        for e in x.a.iter_mut() {
		*e = self.activation.f(*e);
	}

        let mut d_w = gradient.mul_tr(&x);

	//regulate
	for (i, e) in d_w.a.iter_mut().enumerate() {
	    *e = *e + self.w[i] * alpha;
	}
 
        let mut gradient_buf = gradient_buf.lock().unwrap();

        gradient_buf.b.add_by(&gradient);
        gradient_buf.w.add_by(&d_w);

        out_gradient
    }
}
