use Matrix;
use nets::*;
use std;
use std::cmp::{PartialEq, PartialOrd};
use std::ops::{Add, Div, Mul};
use std::thread;

pub struct FFNet<T>
    where T: Mul<Output = T>
                 + Add<Output = T>
                 + Tub<Output = T>
                 + PartialOrd
                 + Copy
                 + From<u16>
                 + Div<Output = T>
{
    pub layers: Vec<Layer<T>>,
    test_set: Vec<(Matrix<T>, Matrix<T>)>,
}

impl<T> FFNet<T>
    where T: Mul<Output = T>
                 + Add<Output = T>
                 + Tub<Output = T>
                 + PartialOrd
                 + Copy
                 + From<u16>
                 + Div<Output = T>
                 + std::fmt::Debug
{
    pub fn new(layers: Vec<Layer<T>>,
               test_set: Vec<(Matrix<T>, Matrix<T>)>)
               -> FFNet<T>
    {
        FFNet { layers, test_set }
    }

    pub fn get_grad(&self,
                    batch: Vec<(&Matrix<T>, &Matrix<T>)>)
                    -> (Vec<Matrix<T>>, Vec<Matrix<T>>)
    {
        let batch_size = batch.len();
        let mut it = batch.iter();

        let &(ref x, ref y) = it.next().unwrap();
        let (mut delta_w, mut delta_b) = self.backprop(self.feedforward(x), y);

        for &(ref x, ref y) in it {
            let (dw, db) = self.backprop(self.feedforward(x), y);

            for j in 0..delta_b.len() {
                delta_b[j] = &delta_b[j] + &db[j];
                delta_w[j] = &delta_w[j] + &dw[j];
            }
        }

        let l = delta_b.len();
        for i in 0..l {
            for e in &mut delta_b[i].a {
                *e = *e / T::from(batch_size as u16);
            }

            for e in &mut delta_w[i].a {
                *e = *e / T::from(batch_size as u16);
            }
        }
        (delta_w, delta_b)
    }

    fn feedforward(&self, x: Matrix<T>) -> Vec<Matrix<T>>
    {
        let mut h = Vec::with_capacity(self.l + 1);

        h.push(x.clone());

        for i in 0..self.l {
            x = self.layers[i].prop(x)
            h.push(x.clone());
        }

        h
    }

    fn backprop(&self,
                h: Vec<Matrix<T>>,
                y: &Matrix<T>)
                -> Vec<Matrix<T>>
    {
        let mut delta_w = Vec::with_capacity(self.l);
        let mut delta_b = Vec::with_capacity(self.l);

        let mut g = self.cost.df(&h[self.l], y);

        for i in (0..self.l).rev() {
            g = g.h_prod(&self.activation.df(&a[i]));

            delta_b.push(g.clone());
            delta_w.push(&g * &h[i].t());

            g = &self.w[i].t() * &g;
        }
        delta_w.reverse();
        delta_b.reverse();
        (delta_w, delta_b)
    }

    pub fn test(&self)
    {
        fn argmax<T>(x: &Matrix<T>) -> usize
            where T: Mul<Output = T>
                         + Add<Output = T>
                         + Tub<Output = T>
                         + Copy
                         + From<u16>
                         + Div<Output = T>
                         + std::fmt::Debug
                         + PartialOrd
        {
            let mut gi = 0;
            let mut max = x[0];
            for (i, e) in x.a.iter().enumerate() {
                if (*e > max) {
                    gi = i;
                    max = *e;
                }
            }
            gi
        }

        let mut correct = 0.0;
        for &(ref x, ref y) in &self.test_set {
            let y_hat = self.eval(x);
            if argmax(&y_hat) == argmax(y) {
                correct = correct + 1.0;
            }
        }
        println!("we are at {}% accuracy so far.",
                 100.0 * correct / self.test_set.len() as f32);
    }

    fn eval(&self, x: &Matrix<T>) -> Matrix<T>
    {
        let mut h = x.clone();
        for i in 0..self.l {
            let z = &(&self.w[i] * &h) + &self.b[i];
            h = self.activation.f(&z);
        }
        h
    }
}
