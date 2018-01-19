use std::ops::{Add, Index, Mul, Sub};
use std::cmp::PartialOrd;

extern crate rand;
use self::rand::Rng;
use std::time::*;

#[derive(Debug)]
pub struct Matrix<T>
    where T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy
{
    pub dim: (usize, usize),
    pub a: Vec<T>,
}

impl<T> Matrix<T>
    where T: Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Copy
        + rand::distributions::range::SampleRange
        + PartialOrd
{
    pub fn new(dim: (usize, usize)) -> Matrix<T>
    {
        Matrix { a: Vec::new(), dim }
    }

    pub fn with_vec(dim: (usize, usize), v: Vec<T>) -> Matrix<T>
    {
        let (m, n) = dim;
        if m * n != v.len() {
            panic!("attempted to set vec of non compatable dimension.")
        }

        Matrix { dim, a: v }
    }

    pub fn new_rand(dim:(usize, usize), low: T, high: T) -> Matrix<T>
    {
        let mut v = Vec::with_capacity(dim.0 * dim.1);
        let mut rng = rand::thread_rng();
        for _ in 0..(dim.0 * dim.1) {
            v.push(rng.gen_range(low, high));
        }
        Matrix::with_vec(dim, v)
    }

    pub fn new_const(dim:(usize, usize), c: T) -> Matrix<T>
    {
        let mut v = Vec::with_capacity(dim.0 * dim.1);
        for _ in 0..(dim.0 * dim.1) {
            v.push(c);
        }
        Matrix::with_vec(dim, v)
    }

    pub fn add_by(&mut self, rhs: &Matrix<T>)
    {
        assert!(self.dim.0 == rhs.dim.0 && self.dim.1 == rhs.dim.1,
                "error: cant add matricies. dimensions dont match");

        let size = self.dim.0 * self.dim.1;
        for i in 0..size {
            self.a[i] = self.a[i] + rhs.a[i];
        }
    }

    pub fn h_prod(&self, rhs: &Matrix<T>) -> Matrix<T>
    {
        assert!(
            self.dim.0 == rhs.dim.0 && self.dim.1 == rhs.dim.1,
            "matrix dimensions are not compatable: self: {:?}, other: {:?}",
            self.dim,
            rhs.dim
        );

        let mut r = Matrix::new(self.dim);

        for i in 0..self.a.len() {
            r.a.push(self.a[i] * rhs.a[i]);
        }
        r
    }

    pub fn t(&self) -> Matrix<T>
    {
        let (m, n) = self.dim;
        let mut v = Vec::with_capacity(m * n);

        for i in 0..n {
            for j in 0..m {
                v.push(self[j * n + i])
            }
        }
        Matrix::with_vec((n, m), v)
    }

    pub fn len(&self) -> usize
    {
        self.a.len()
    }

    // equivalent to: self * (rhs^t)
    pub fn mul_tr(&self, rhs: &Self) -> Matrix<T>
    {
        let (m, n) = self.dim;
        let (m1, n1) = rhs.dim;

        assert!(n == n1,
                "can't do self*(rhs^t) with dimensions, {:?} and {:?}",
                self.dim,
                rhs.dim);

        let mut r = Matrix::new((m, m1));

        let mut acc;
        for i in 0..m {
            for j in 0..m1 {
                acc = rhs[j * n1] * self[i * n];
                for k in 1..n {
                    acc = acc + rhs[j * n1 + k] * self[i * n + k];
                }
                r.a[i * m1 + j] = acc;
            }
        }
        r
    }

    // equivalent to: (self^t) * rhs
    pub fn mul_tl(&self, rhs: &Self) -> Matrix<T>
    {
        let (m, n) = self.dim;
        let (m1, n1) = rhs.dim;

        assert!(m == m1,
                "can't do (self^t)*rhs with dimensions, {:?} and {:?}",
                self.dim,
                rhs.dim);

        let mut r = Matrix::new((n, n1));

        let mut acc;
        for i in 0..n {
            for j in 0..n1 {
                acc = rhs[j] * self[i];
                for k in 1..m {
                    acc = acc + rhs[j + k * n1] * self[i + n * k];
                }
                r.a[i * n1 + j] = acc;
            }
        }
        r
    }
}

impl<T> Clone for Matrix<T>
    where T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy
            + rand::distributions::range::SampleRange
            + PartialOrd
{
    fn clone(&self) -> Self
    {
        let mut a1 = Matrix::new(self.dim);
        for e in &self.a {
            a1.a.push(*e);
        }
        a1
    }
}

impl<'a, 'b, T> Mul<&'b Matrix<T>> for &'a Matrix<T>
    where T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy
        + rand::distributions::range::SampleRange
        + PartialOrd
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &'b Matrix<T>) -> Matrix<T>
    {
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
    where T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy
        + rand::distributions::range::SampleRange
        + PartialOrd
{
    type Output = Matrix<T>;

    fn mul(self, rhs: T) -> Matrix<T>
    {
        Matrix::with_vec(self.dim, self.a.iter().map(|e| *e * rhs).collect())
    }
}

impl<'a, 'b, T> Add<&'b Matrix<T>> for &'a Matrix<T>
    where T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy
        + rand::distributions::range::SampleRange
        + PartialOrd
{
    type Output = Matrix<T>;

    fn add(self, rhs: &'b Matrix<T>) -> Matrix<T>
    {
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
    where T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy
        + rand::distributions::range::SampleRange
        + PartialOrd
{
    type Output = Matrix<T>;

    fn sub(self, rhs: &'b Matrix<T>) -> Matrix<T>
    {
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
    where T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy
{
    type Output = T;

    fn index(&self, i: usize) -> &T
    {
        &self.a[i]
    }
}
