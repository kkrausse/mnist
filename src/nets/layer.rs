use nets::*;

struct Layer<T>
    where T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy
{
    w: Matrix<T>,
    b: Matrix<T>,
    activation: Activation<T>,
}

impl Layer<T>
    where T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy
{
    pub fn new_rand(activation: Activation<T>, into: usize, out: usize)
                    -> Layer
    {
        Layer { w: Matrix::new_rand((out, into), -0.05, 0.05),
                b: Matrix::new_const((out, 1), 0.01),
                activation, }
    }

    pub fn prop(&self, x: Matrix<T>) -> Matrix<T>
    {
        x.a.iter_mut().for_each(|e| *e = self.activation.f(*e));
        let mut out = &w * &x;
        out.add(&b);
        out
    }

    /*
     * TODO: Add regularization terms for layer parameters.
     *	And do ARC so i can update the gradient_buf in multiple threads.
     */
    pub fn backprop(&self,
                    gradient: Matrix<T>,
                    x: Matrix<T>,
                    gradient_buf: &mut Self)
                    -> Matrix<T>
    {
        let mut out_gradient = self.w.mul_tl(&gradient); //same as (w^t)*g

        //hadamar product of g with df with respect to the input.
        for i in out_gradient.len() {
            out_gradient.a[i] = out_gradient[i] * x[i]
        }

        //recompute f(x)
        x.a.iter_mut().for_each(|e| *e = self.activation.f(*e));

        //add regularization term at some point too..
        let d_w = g.mul_tr(&x);

        gradient_buf.b.add_by(&gradient);
        gradient_buf.w.add_by(d_w);
    }
}
