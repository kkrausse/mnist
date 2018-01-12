use ::Matrix;
use ::nets::*;

use std::ops::{Add, Mul, Div};
use std::cmp::{PartialOrd, PartialEq};

use std::thread;
use std;

pub struct FFNet <S, T, U>
	where S: Mul<Output=S> + Add<Output=S> + Sub<Output=S> + PartialOrd +
			Copy + From<u16> + Div<Output=S>,
		T: Activation<S>,
		U: Cost<S>,
{
	pub w: Vec<Matrix<S>>,
	pub b: Vec<Matrix<S>>,
	l: usize,
	activation: T,
	cost: U,
	test_set: Vec<(Matrix<S>, Matrix<S>)>,
}

impl<S, T, U> FFNet<S, T, U>
	where 
		S: Mul<Output=S> + Add<Output=S> + Sub<Output=S> + PartialOrd +
			Copy + From<u16> + Div<Output=S> + std::fmt::Debug,
		T: Activation<S>,
		U: Cost<S>,
{
	pub fn new(w: Vec<Matrix<S>>, b: Vec<Matrix<S>>, activation: T, 
		cost: U, test_set: Vec<(Matrix<S>, Matrix<S>)>) -> FFNet<S, T, U> {

		if w.len() != b.len() { panic!("biases and weight vecs should be same dimension") }
		let l = b.len();
		FFNet {
			w,
			b,
			l,
			activation,
			cost,
			test_set,
		}
	}

	pub fn get_grad(&self, batch: Vec<(&Matrix<S>, &Matrix<S>)>) 
		-> (Vec<Matrix<S>>, Vec<Matrix<S>>) {
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
				*e = *e / S::from(batch_size as u16);
			}

			for e in &mut delta_w[i].a {
				*e = *e / S::from(batch_size as u16);
			}
		}
		(delta_w, delta_b)
	}

	fn feedforward(&self, x: &Matrix<S>) -> (Vec<Matrix<S>>, Vec<Matrix<S>>) {
		let mut h = Vec::with_capacity(self.l + 1);
		let mut a = Vec::with_capacity(self.l);

		h.push( x.clone() );
		for i in 0..self.l {
			a.push( &(&self.w[i] * &h[i]) + &self.b[i] );
			h.push( self.activation.f(&a[i]) );
		}
		(h, a)
	}

	fn backprop(&self, (h, a): (Vec<Matrix<S>>, Vec<Matrix<S>>), y: &Matrix<S>) 
		-> (Vec<Matrix<S>>, Vec<Matrix<S>>) {

		let mut delta_w = Vec::with_capacity(self.l);
		let mut delta_b = Vec::with_capacity(self.l);

		let mut g = self.cost.df(&h[self.l], y);

		for i in (0..self.l).rev() {
			g = g.h_prod( &self.activation.df(&a[i]));

			delta_b.push(g.clone());
			delta_w.push(&g * &h[i].t());

			g = &self.w[i].t() * &g;
		}
		delta_w.reverse();
		delta_b.reverse();
		(delta_w, delta_b)
	}

	pub fn test(&self) {
		fn argmax<S>(x: &Matrix<S>) -> usize
			where S: Mul<Output=S> + Add<Output=S> + Sub<Output=S> +
			 Copy + From<u16> + Div<Output=S> + std::fmt::Debug + PartialOrd
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
			if argmax(&y_hat) == argmax(y) {correct = correct + 1.0;}
		}
		println!("we are at {}% accuracy so far.", 100.0 * correct / self.test_set.len() as f32);
	}

	fn eval(&self, x: &Matrix<S>) -> Matrix<S> {
		let mut h = x.clone();
		for i in 0..self.l {
			let z = &(&self.w[i] * &h) + &self.b[i];
			h = self.activation.f(&z);
		}
		h
	}
}
