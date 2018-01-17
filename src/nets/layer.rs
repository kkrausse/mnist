use ::nets::*;

//type Number<T> where T: Mul<Output=T> + Add<Output=T> + Sub<Output=T> + Copy = T;

struct Layer<T>
	where T: Mul<Output=T> + Add<Output=T> + Sub<Output=T> + Copy
{
	w: Matrix<T>,
	b: Matrix<T>,
	activation: Activation<T>,
}

impl Layer<T> 
	where T: Mul<Output=T> + Add<Output=T> + Sub<Output=T> + Copy
{
	pub fn new_rand(activation: Activation<T>, into: usize, out: usize)
		-> Layer
	{
		Layer {
			w: Matrix::new_rand((out, into), -0.05, 0.05),
			b: Matrix::new_const((out, 1), 0.01),
			activation,
		}
	}

	pub fn prop(&self, x: Matrix<T>) -> Matrix<T> {
		x.a.iter_mut().for_each(|e| *e = self.activation.f(*e));
		let out = &w * &x;
		out.add(&b);
		out
	}

	pub fn backprop(&self, g: Matrix<T>) -> Matrix<T> {
		
	}
}