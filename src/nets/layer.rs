use nets::*;
use std::sync::Mutex;

pub struct Layer<A: Activation<Number>>
{
    w: Matrix<Number>,
    b: Matrix<Number>,
    activation: A,
}

impl<A: Activation<Number>> Layer<A>
{
    pub fn new_rand(activation: A, into: usize, out: usize)
                    -> Layer<A>
    {
        Layer { w: Matrix::new_rand((out, into), -0.05, 0.05),
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
        x.a.iter_mut().for_each(|e| *e = self.activation.f(*e));
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
        let mut out_gradient = self.w.mul_tl(&gradient); //same as (w^t)*g

        //hadamar product of g with df with respect to the input.
        for i in 0..out_gradient.len() {
            out_gradient.a[i] = out_gradient[i] * x[i]
        }

        //recompute f(x)
        x.a.iter_mut().for_each(|e| *e = self.activation.f(*e));

        //add regularization term at some point too..
        let d_w = gradient.mul_tr(&x);

        let mut gradient_buf = gradient_buf.lock().unwrap();

        gradient_buf.b.add_by(&gradient);
        gradient_buf.w.add_by(&d_w);

        out_gradient
    }
}
