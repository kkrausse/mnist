use std::ops::{Add, Sub, Mul, Index};

#[derive(Debug)]
pub struct Matrix <T>
	where T: Mul<Output=T> + Add<Output=T> + Sub<Output=T> + Copy 
{
	pub dim: (usize, usize),
	pub a: Vec<T>,
}

impl<T> Matrix<T> 
	where T: Mul<Output=T> + Add<Output=T> + Sub<Output=T> + Copy
{
	pub fn new(dim: (usize, usize)) -> Matrix<T> {
		Matrix {
			a: Vec::new(),
			dim, 
		}
	}

	pub fn with_vec(dim: (usize, usize), v: Vec<T>) -> Matrix<T> {
		let (m,n) = dim;
		if m * n !=  v.len() {
			panic!("attempted to set vec of non compatable dimension.")
		}

		Matrix {
			dim,
			a: v,
		}
	}

	pub fn h_prod(&self, rhs: &Matrix<T>) -> Matrix<T> {
		let (m1, n1) = rhs.dim;
		let (m, n) = self.dim;

		if m1 != m || n1 != n {
			panic!("matrix dimensions are not compatable: self: {:?}, other: {:?}", self.dim, rhs.dim);
		}

		let mut r = Matrix::new(self.dim);

		for i in 0..self.a.len() {
			r.a.push(self.a[i] * rhs.a[i]);
		}
		r
	}

	pub fn t(&self) -> Matrix<T> {
		let (m, n) = self.dim;
		let mut v = Vec::with_capacity(m * n);

		for i in 0..n {
			for j in 0..m {
				v.push( self[j * n + i] )
			}
		}
		Matrix::with_vec((n, m), v)
	}

	pub fn len(&self) -> usize {self.a.len()}
}

impl<T> Clone for Matrix<T> 
	where T: Mul<Output=T> + Add<Output=T> + Sub<Output=T> + Copy
{
	fn clone(&self) -> Self {
		let mut a1 = Matrix::new(self.dim);
		for e in &self.a {
			a1.a.push(*e);
		}
		a1
	}
}

impl<'a,'b, T> Mul<&'b Matrix<T>> for &'a Matrix<T>
	where T: Mul<Output=T> + Add<Output=T> + Sub<Output=T> + Copy
{
	type Output = Matrix<T>;

	fn mul(self, rhs: &'b Matrix<T>) -> Matrix<T> {
		let (m1, n1) = rhs.dim;
		let (m, n) = self.dim;

		if n != m1 {
			panic!("matrix dimensions are not compatable.");
		}

		let mut r = Matrix::new((m, n1));

		for i in 0..m {
			for j in 0..n1 {
				let mut acc = self.a[i * n] * rhs.a[j];
				for k in 1..n {
					acc = acc + self.a[i * n + k] * rhs.a[j + k * n1];
				}
				r.a.push(acc);
			}
		}
		r
	}
}

impl<'a, T> Mul<T> for &'a Matrix<T>
	where T: Mul<Output=T> + Add<Output=T> + Sub<Output=T> + Copy
{
	type Output = Matrix<T>;

	fn mul(self, rhs: T) -> Matrix<T> {
		Matrix::with_vec(self.dim, self.a.iter().map(|e| *e * rhs).collect())
	}
}


impl<'a, 'b, T> Add<&'b Matrix<T>> for &'a Matrix<T>
	where T: Mul<Output=T> + Add<Output=T> + Sub<Output=T> + Copy
{
	type Output = Matrix<T>;

	fn add(self, rhs: &'b Matrix<T>) -> Matrix<T> {
		let (m1, n1) = rhs.dim;
		let (m, n) = self.dim;

		if m1 != m || n1 != n {
			panic!("matrix dimensions are not compatable");
		}

		let mut r = Matrix::new(self.dim);

		for i in 0..self.a.len() {
			r.a.push(self.a[i] + rhs.a[i]);
		}
		r
	}
}

impl<'a, 'b, T> Sub<&'b Matrix<T>> for &'a Matrix<T>
	where T: Mul<Output=T> + Add<Output=T> + Sub<Output=T> + Copy
{
	type Output = Matrix<T>;

	fn sub(self, rhs: &'b Matrix<T>) -> Matrix<T> {
		let (m1, n1) = rhs.dim;
		let (m, n) = self.dim;

		if m1 != m || n1 != n {
			panic!("matrix dimensions are not compatable");
		}

		let mut r = Matrix::new(self.dim);

		for i in 0..self.a.len() {
			r.a.push(self.a[i] - rhs.a[i]);
		}
		r
	}
}

impl<T> Index<usize> for Matrix<T>
	where T: Mul<Output=T> + Add<Output=T> + Sub<Output=T> + Copy
{
	type Output = T;

	fn index(&self, i: usize) -> &T {&self.a[i]}
}