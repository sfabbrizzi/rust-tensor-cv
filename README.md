# RustCV

The aim of this project is to create a Rust library for computer vision tasks. This starts as a personal learn-by-doing project that first implements a tensor library (following the tutorial at [this link](https://huggingface.co/blog/KeighBee/tensors-from-scratch-in-rust-p1)) and then builds on top of it to implement various computer vision algorithms and techniques.

## Features

- [ ] Tensor library
    - [x] Tensor struct
    - [x] indexing
    - [ ] view operations
        - [x] permute
        - [x] merge
        - [x] split
        - [ ] reshape
        - [ ] slice
        - [ ] skip
    - [ ] data operations
- [ ] Image processing and manipulation
- [ ] Feature detection and matching
- [ ] Object detection and recognition

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rustcv = "0.1"
```

## Quick Start

```rust
use rust_tensor_cv::core::Tensor;

fn main() {
    let tensor = Tensor::<f32>::zeroes(vec![64, 64, 3]); 
    println!("Tensor: {:?}", tensor);
}
```

## Documentation

For detailed documentation, run:

```bash
cargo doc --open
```

## Tests

To run the tests, run:

```bash
cargo test
```

## License

MIT