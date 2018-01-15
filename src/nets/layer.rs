use ::nets::*;

//type Number<T> where T: Mul<Output=T> + Add<Output=T> + Sub<Output=T> + Copy = T;

struct Layer<T>
	where: T: Mul<Output=T> + Add<Output=T> + Sub<Output=T> + Copy
{
	w: Matrix<T>,
	b: Matrix<T>,
	activation: Activation<T>,
}

impl Layer<T> 
	where: T: Mul<Output=T> + Add<Output=T> + Sub<Output=T> + Copy
{
	pub fn new_rand(activation: Activation<Number>, into: usize, out: usize)
		-> Layer
	{
		Layer {
			w: Matrix::new_rand((out, into), -0.01, 0.01),
			b: Matrix::new_const((out, 1), 0.01),
			activation,
		}
	}

	pub fn prop(&self, x: Matrix<T>) -> Matrix<T> {
		//need to fix this so there aren't so many needless copies.
		let out = &w * &self.activation.f(&x);
		out.add(&b);
		out
	}

	// pub fn backprop(&self, g: Matrix<T>) -> Matrix<T> {
		
	// }
}