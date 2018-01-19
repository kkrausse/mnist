#![allow(dead_code)]

extern crate mnist;
use mnist::nets::*;

use mnist::Matrix;

extern crate crossbeam;
//use crossbeam;

use std::fs::File;
use std::io::Read;

fn main()
{
    let xs = read_idx("./res/train-images.idx3-ubyte", 2_000);
    println!("got xs");
    let ys = read_idx("./res/train-labels.idx1-ubyte", 2_000);
    println!("got ys");

    let mut tx = read_idx("./res/train-images.idx3-ubyte", 200);
    let mut ty = read_idx("./res/train-labels.idx1-ubyte", 200);
    println!("got test sets");

    let tlen = tx.len();
    let test_set:Vec<_> = tx.drain(0..tlen).zip(ty.drain(0..tlen)).collect();

    //let l = 15;
    let num_cores = 2;
    let step = 0.08;
    let batch_size = 40;
    //let step_decay = 0.96;

    let af = ATan{};

    let mut net = FFNet::new(
            vec![
                Layer::new_rand(af.clone(), 28 * 28, 20 * 20),
                Layer::new_rand(af.clone(), 20 * 20, 100),
                Layer::new_rand(af.clone(), 100, 100),
                Layer::new_rand(af.clone(), 100, 10)
            ],
            test_set,
            num_cores);

    net.train(batch_size, step, (xs, ys));
}

fn read_idx(fname: &str, num_vals: usize) -> Vec<Matrix<f32>>
{
    let f = File::open(fname).unwrap();
    let mut bytes = f.bytes();

    let magic_num = read_int(&mut bytes);
    let dim = magic_num % (1 << 3);
    let len = read_int(&mut bytes);

    return match dim {
        1 => {
            let mut r = Vec::with_capacity(len);
            while let Some(res) = bytes.next() {
                if let Ok(b) = res {
                    let mut v = vec![0.0; 10];
                    v[b as usize] = 1.0;
                    r.push(Matrix::with_vec((10, 1), v));
                    if r.len() >= num_vals {
                        break;
                    }
                } else {
                    panic!("some error with reading the byte from the file");
                }
            }
            r
        }
        3 => {
            let (m, n) = (read_int(&mut bytes), read_int(&mut bytes));
            let mut r = Vec::with_capacity(len);

            let mut i = 0;
            let mut v = Vec::with_capacity(m * n);

            while let Some(res) = bytes.next() {
                if !(i < m * n) {
                    i = 0;
                    r.push(Matrix::with_vec((m * n, 1), v));
                    v = Vec::with_capacity(m * n);
                    if r.len() >= num_vals {
                        break;
                    }
                }
                if let Ok(b) = res {
                    v.push(b as f32);
                } else {
                    panic!("some error with reading the byte from the file");
                }
                i += 1;
            }
            r
        }
        _ => panic!("some problem with the dimension read from the idx file"),
    };

    fn print_num(m: &Matrix<f32>)
    {
        for (i, d) in m.a.iter().enumerate() {
            if i % m.dim.0 == 0 {
                println!("");
            }
            if *d > 200.0 {
                print!("#");
            } else if *d > 100.0 {
                print!("o");
            } else if *d > 0.0 {
                print!(".");
            } else {
                print!(" ");
            }
        }
    }

    fn read_int(it: &mut std::io::Bytes<std::fs::File>) -> usize
    {
        let mut n: usize = 0;
        for i in 0..4 {
            n += (it.next().unwrap().unwrap() as usize) << ((3 - i) * 8);
        }
        n
    }
}

// fn d_2_sec(t: Duration) -> f64
// {
//     let ns = (t.subsec_nanos() / 1000) as f64;
//     t.as_secs() as f64 + ns / 1_000_000.0
// }

// fn const_mat(m: usize, n: usize, c: f32) -> Matrix<f32>
// {
//     let mut a = Vec::with_capacity(m * n);
//     for _ in 0..(m * n) {
//         a.push(c);
//     }
//     Matrix::with_vec((m, n), a)
// }

// fn rand_mat(m: usize, n: usize) -> Matrix<f32>
// {
//     Matrix::with_vec((m, n), gen_vec(m * n))
// }

// fn gen_vec(n: usize) -> Vec<f32>
// {
//     let mut r = Vec::with_capacity(n);
//     let mut rng = rand::thread_rng();
//     for _ in 0..n {
//         r.push(rng.gen_range(-0.05, 0.05));
//     }
//     r
// }
