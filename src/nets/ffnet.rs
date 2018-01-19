//use Matrix;
use nets::*;
use std;
// use std::cmp::PartialOrd;
// use std::ops::{Add, Div, Mul};
use std::sync::{Arc, Mutex};
use ::thread_pool::ThreadPool;


extern crate crossbeam;

/*
    these are the type params from before:
        Mul<Output = T>
         + Add<Output = T>
         + Sub<Output = T>
         + PartialOrd
         + Copy
         + From<u16>
         + Div<Output = T>
         + rand::distributions::range::SampleRange
         + std::marker::Send
         + std::marker::Sync

    this is just unmanagable so im switching to a type alias
*/

pub struct FFNet
{
    pub layers: Vec<Layer<AFunc>>,
    l: usize,
    test_set: Vec<(Matrix<Number>, Matrix<Number>)>,
    grad_buf: Vec<Mutex<Layer<AFunc>>>,
    num_threads: usize,
    output: OutFunc,
}

impl FFNet
{
    pub fn new(layers: Vec<Layer<AFunc>>,
               test_set: Vec<(Matrix<Number>, Matrix<Number>)>,
               num_threads: usize)
               -> FFNet
    {
        let l = layers.len();
        let mut grad_buf = Vec::with_capacity(l);
        for layer in &layers {
            grad_buf.push(Mutex::new(layer.clone_zeros()));
        }
        let output = OutFunc{};

        FFNet { layers, 
                l, 
                test_set,
                grad_buf,
                num_threads,
                output}
    }

    pub fn train(&mut self,
                batch_size: usize,
                step: Number,
                (x, y): (Vec<Matrix<Number>>, Vec<Matrix<Number>>))
    {
        let mut i = 0;
        loop {

            let mut batch = Vec::with_capacity(batch_size);

            let mut rng = rand::thread_rng();
            for _ in 0..batch_size {
                let rando = rng.gen::<usize>() % x.len();
                batch.push((&x[rando], &y[rando]));
            }

            self.update_with_batch(batch);



            println!("did batch");
            if i % 6 == 0 {
                self.test();
            }
            i = i + 1;
        }

    }

    pub fn update_with_batch(&self,
                    mut batch: Vec<(&Matrix<Number>, &Matrix<Number>)>)
    {
        let batch_size = batch.len();

        let self_arc = Arc::new(self);

        let props_per_thread = batch_size / self.num_threads;
        let chunks = batch.chunks(props_per_thread);

        crossbeam::scope(|scope| {
            for chunk in chunks {
                let netref = self;
                scope.spawn(move || {
                    for &(x, y) in chunk {
                        let (x, y) = (x.clone(), y.clone());
                        FFNet::add_to_gradient(netref, x, y);
                    }
                });
            }
        });
    }

    pub fn update_params(&mut self, batch_size: usize, step) {
        unimplemented!()
    }

    fn add_to_gradient(net: &FFNet,
                        x: Matrix<Number>,
                        y: Matrix<Number>)
    {
        let h = net.prop(x);
        net.backprop(h, &y);
    }

    fn prop(&self, mut x: Matrix<Number>) -> Vec<Matrix<Number>>
    {
        let mut h = Vec::with_capacity(self.l + 1);

        h.push(x.clone());

        for layer in &self.layers {
            x = layer.prop(x);
            h.push(x.clone());
        }

        h.push(self.output.f(x));
        h
    }

    fn backprop(&self,
                mut h: Vec<Matrix<Number>>,
                y: &Matrix<Number>)
    {
        let mut gradient = self.output.df(&h.pop().unwrap(), y);

        for i in (0..self.l).rev() {
            gradient = self.layers[i]
                        .backprop(gradient, h.pop().unwrap(), &self.grad_buf[i]);
        }
    }

    pub fn test(&self)
    {
        fn argmax(x: &Matrix<Number>) -> usize
        {
            let mut gi = 0;
            let mut max = x[0];
            for (i, e) in x.a.iter().enumerate() {
                if *e > max {
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

    fn eval(&self, x: &Matrix<Number>) -> Matrix<Number>
    {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.prop(x);
        }

        self.output.f(x)
    }
}
