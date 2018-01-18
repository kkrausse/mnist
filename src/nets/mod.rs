pub use self::ffnet::FFNet;
pub mod ffnet;

pub use self::layer::Layer;
pub mod layer;

use Matrix;
use std::cmp::PartialEq;
use std::f32;
use std::ops::{Add, Mul, Sub};

pub trait Activation<T>
    where T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy
{
    fn f(&self, x: T) -> T;
    fn df(&self, x: T) -> T;
}

pub trait Cost<T>
    where T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy
{
    type Domain;
    fn f(&self, y_hat: &Matrix<T>, y: &Matrix<T>) -> Matrix<T>;
    fn df(&self, y_hat: &Matrix<T>, y: &Matrix<T>) -> Matrix<T>;
}

pub struct RLU {}

impl Activation<f32> for RLU
{
    type Domain = Matrix<f32>;

    fn f(&self, x: &Matrix<f32>) -> Matrix<f32>
    {
        let v: Vec<_> = x.a.iter()
                         .map(|e| if *e > 0.0 { *e } else { 0.0 })
                         .collect();
        Matrix::with_vec(x.dim, v)
    }

    fn df(&self, x: &Matrix<f32>) -> Matrix<f32>
    {
        let v: Vec<_> = x.a.iter()
                         .map(|e| if *e >= 0.0 { 1.0 } else { 0.0 })
                         .collect();
        Matrix::with_vec(x.dim, v)
    }
}

pub struct ATan {}

impl Activation<f32> for ATan
{
    fn f(&self, x: f32) -> f32
    {
        f32::atan(x)
    }

    fn df(&self, x: &Matrix<f32>) -> Matrix<f32>
    {
        1.0 / (1.0 + x * x)
    }
}

// pub struct CrossEntropy {}

// impl<f32> Cost<f32> for CrossEntropy {
// 	fn f(&self, y_hat: &Matrix<f32>, y: &Matrix<f32>) -> Matrix<f32> {
// 		let (m, n) = y_hat.dim;
// 		let (m1, n1) = y.dim;
// 		if m1 != m || n1 != n || n != 1 {
// 			panic!("incompatable matrix dimenstions for cost func");
// 		}
// 		let fun = |y, yh| {
// 			y * f32::ln(yh) + (1.0 - y) * f32::ln(1.0 - yh);
// 		};

// 		let v: Vec<_> = y_hat.a.iter().zip(y.a.iter()).map( fun ).collect();

// 		Matrix::with_vec(y.dim, v)
// 	}

// 	fn df(&self, y_hat: &Matrix<T>, y: &Matrix<T>) -> Matrix<T> {
// 		let (m, n) = y_hat.dim;
// 		let (m1, n1) = y.dim;
// 		if m1 != m || n1 != n || n != 1 {
// 			panic!("incompatable matrix dimenstions for cost func");
// 		}

// 		let fun = |y, yh| {
// 			y * f32::ln(yh) + (1.0 - y) * f32::ln(1.0 - yh);
// 		};

// 		let v: Vec<_> = y_hat.a.iter().zip(y.a.iter()).map( fun ).collect();

// 		Matrix::with_vec(y.dim, v)
// 	}
// }

pub struct MeanSquareError {}

impl<T> Cost<T> for MeanSquareError
    where T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy
{
    type Domain = Matrix<T>;

    fn f(&self, y_hat: &Matrix<T>, y: &Matrix<T>) -> Matrix<T>
    {
        let (m, n) = y_hat.dim;
        let (m1, n1) = y.dim;
        if m1 != m || n1 != n || n != 1 {
            panic!("incompatable matrix dimenstions for cost func");
        }

        let v: Vec<_> = y_hat.a
                             .iter()
                             .zip(y.a.iter())
                             .map(|(y, yh)| (*y - *yh) * (*y - *yh))
                             .collect();

        Matrix::with_vec(y.dim, v)
    }

    fn df(&self, y_hat: &Matrix<T>, y: &Matrix<T>) -> Matrix<T>
    {
        let (m, n) = y_hat.dim;
        let (m1, n1) = y.dim;
        if m1 != m || n1 != n || n != 1 {
            panic!("incompatable matrix dimenstions for cost func");
        }

        let mut r = Matrix::new(y.dim);

        for i in 0..y.a.len() {
            r.a.push(y_hat[i] - y[i]);
        }
        r
    }
}
