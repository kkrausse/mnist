pub use matrix::Matrix;
pub mod matrix;
pub mod nets;

#[cfg(test)]
mod tests
{
    use matrix;
    #[test]
    fn it_works()
    {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn rusty_trombone()
    {
        let mut a = matrix::Matrix::with_vec((10, 1), vec![2.0; 10]);

        println!("{:?}", a);
    }
}
