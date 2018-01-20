pub use self::ffnet::FFNet;
pub mod ffnet;

pub use self::layer::Layer;
pub mod layer;

use Matrix;

use std::cmp::PartialEq;
use std::f32;
use std::ops::{Add, Mul, Sub};

extern crate rand;
use self::rand::Rng;
use std::time::*;

pub type Number = f32;

pub type OutFunc = Softmax;
pub type AFunc = RELU;


pub trait Activation<T>
    where T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy
{
    fn f(&self, x: T) -> T;
    fn df(&self, x: T) -> T;
    fn clone(&self) -> Self;
}

pub trait Output<T>
    where T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy
{
    fn f(&self, x: Matrix<T>) -> Matrix<T>;
    fn df(&self, y_hat: &Matrix<T>, y: &Matrix<T>) -> Matrix<T>;
}

pub struct ATan {}

impl Activation<f32> for ATan
{
    fn f(&self, x: f32) -> f32
    {
        f32::atan(x)
    }

    fn df(&self, x: f32) -> f32
    {
        1.0 / (1.0 + x * x)
    }

    fn clone(&self) -> Self
    {
        ATan{}
    }
}

pub struct RELU {}

impl Activation<Number> for RELU
{
    fn f(&self, x: f32) -> f32
    {
        if x > 0.0 { x } else { 0.0 }
    }

    fn df(&self, x: f32) -> f32
    {
        if x >= 0.0 { 1.0 } else { 0.0 }
    }

    fn clone(&self) -> Self
    {
        RELU{}
    }
}


pub struct Softmax{}

impl Output<f32> for Softmax
{
    fn f(&self, x: Matrix<f32>) -> Matrix<f32>
    {
        let mut y = Matrix::new(x.dim);

        let mut exp_sum = 0.0;
        for e in &x.a {
            exp_sum = exp_sum + e.exp();
        }

        for e in x.a {
            y.a.push(e.exp() / exp_sum);
        }
        y
    }

    fn df(&self, y_hat: &Matrix<f32>, y: &Matrix<f32>) -> Matrix<f32>
    {
        y_hat - y
    }
}
// pub struct MeanSquareError {}

// impl<T> Cost<T> for MeanSquareError
//     where T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy
// {
//     type Domain = Matrix<T>;

//     fn f(&self, y_hat: &Matrix<T>, y: &Matrix<T>) -> Matrix<T>
//     {
//         let (m, n) = y_hat.dim;
//         let (m1, n1) = y.dim;
//         if m1 != m || n1 != n || n != 1 {
//             panic!("incompatable matrix dimenstions for cost func");
//         }

//         let v: Vec<_> = y_hat.a
//                              .iter()
//                              .zip(y.a.iter())
//                              .map(|(y, yh)| (*y - *yh) * (*y - *yh))
//                              .collect();

//         Matrix::with_vec(y.dim, v)
//     }

//     fn df(&self, y_hat: &Matrix<T>, y: &Matrix<T>) -> Matrix<T>
//     {
//         let (m, n) = y_hat.dim;
//         let (m1, n1) = y.dim;
//         if m1 != m || n1 != n || n != 1 {
//             panic!("incompatable matrix dimenstions for cost func");
//         }

//         let mut r = Matrix::new(y.dim);

//         for i in 0..y.a.len() {
//             r.a.push(y_hat[i] - y[i]);
//         }
//         r
//     }
// }
