use rust_tensor_cv::core::Tensor;

fn main() {
    let tensor = Tensor::<i32>::ones(vec![2, 2]);
    println!("{:?}", tensor[&[1, 1]]);
}
