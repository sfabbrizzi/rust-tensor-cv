use num_traits::{One, Zero};
use std::ops::{Index, IndexMut, RangeInclusive};

#[derive(Debug, Clone, PartialEq)]
/// A struct representing the shape of a tensor, which is a vector of dimensions.
pub struct TensorShape {
    shape: Vec<usize>,
    strides: Vec<usize>, // added to optimize rveling and unraveling and allow view ops.
    linear_offset: usize,
}

impl TensorShape {
    /// Creates a new TensorShape with the given shape.
    pub fn new(shape: Vec<usize>) -> Self {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        TensorShape { shape, strides, linear_offset: 0 }
    }

    /// Returns the total number of elements in the tensor, which is the product of the dimensions.
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Converts multi-dimensional indices to a single linear index.
    pub fn ravel_index(&self, indices: &[usize]) -> usize {
        if indices.len() != self.shape.len() {
            panic!("Indices needs to match shape")
        }

        indices
            .iter()
            .zip(self.strides.iter())
            .rev()
            .map(|(&idx, &stride)| idx * stride)
            .sum()
    }

    /// Converts a linear index to the corresponding multi-dimensional indices.
    pub fn unravel_index(&self, linear_index: usize) -> Vec<usize> {
        if self.shape.is_empty() {
            return vec![];
        }

        let mut indices: Vec<usize> = vec![0 as usize; self.shape.len()];
        let mut remaining_index = linear_index;

        for (i, &stride) in self.strides.iter().enumerate() {
            indices[i] = remaining_index / stride;
            remaining_index %= stride;
        }

        indices
    }

    /// Permutes the dimensions of the tensor according to the given order of indices.
    pub fn permute(&self, permuted_indices: &[usize]) -> Self {
        let shape = permuted_indices.iter().map(|x| self.shape[*x]).collect();

        let strides = permuted_indices.iter().map(|x| self.strides[*x]).collect();

        TensorShape { shape, strides, linear_offset: self.linear_offset }
    }

    // Merges consecutive dimensions of the tensor.
    pub fn merge(&self, to_merge: RangeInclusive<usize>) -> Self {
        let (start, end) = (*to_merge.start(), *to_merge.end());
        assert!(
            start <= end && end < self.shape.len(),
            "Invalid range for merging dimensions"
        );

        let merged_size = self.shape[to_merge.clone()].iter().product();
        let merged_stride = self.strides[end];

        let mut new_shape = Vec::<usize>::with_capacity(self.shape.len() - (end - start));
        let mut new_strides = Vec::<usize>::with_capacity(self.strides.len() - (end - start));

        new_shape.extend_from_slice(&self.shape[..start]);
        new_shape.push(merged_size);
        new_shape.extend_from_slice(&self.shape[end + 1..]);

        new_strides.extend_from_slice(&self.strides[..start]);
        new_strides.push(merged_stride);
        new_strides.extend_from_slice(&self.strides[end + 1..]);

        TensorShape {
            shape: new_shape,
            strides: new_strides,
            linear_offset: self.linear_offset
        }
    }

    // Splits a dimension into a compatible higher number of dimensions.
    // If 0 is passed into shape it acts as a wildcard and the function
    // computes the right dimension. Only one wildcard can be passed.
    pub fn split(&self, dim: usize, shape: &[usize]) -> Self {
        if dim >= self.shape.len() {
            panic!("Dimension index out of bounds");
        }

        let original_size = self.shape[dim];

        // Calculate the product of non-zero sizes and find wildcard
        let mut non_zero_product = 1usize;
        let mut zero_index = None;

        for (i, &size) in shape.iter().enumerate() {
            if size == 0 {
                if zero_index.is_some() {
                    panic!("Cannot have more than one wildcard (0) in split sizes");
                }
                zero_index = Some(i);
            } else {
                non_zero_product *= size;
            }
        }

        // Create the final sizes, inferring wildcards
        let mut final_sizes = shape.to_vec();
        if let Some(zero_index) = zero_index {
            if original_size % non_zero_product != 0 {
                panic!(
                    "Cannot split dimension of size {} into sizes {:?}",
                    original_size, shape
                );
            }
            let inferred_size = original_size / non_zero_product;
            final_sizes[zero_index] = inferred_size;
        }

        let mut new_shape = Vec::<usize>::with_capacity(self.shape.len() - 1 + shape.len());

        new_shape.extend_from_slice(&self.shape[..dim]);
        new_shape.extend_from_slice(&final_sizes);
        if dim + 1 < self.shape.len() {
            new_shape.extend_from_slice(&self.shape[dim + 1..]);
        }

        TensorShape::new(new_shape)
    }

    pub fn reshape(&self, _shape: &[usize]) -> Self {
        unimplemented!();
    }

    pub fn slice(&self, dim: usize, range: RangeInclusive<usize>) -> Self {
        if dim >= self.shape.len() {
            panic!("Dimension index out of bounds");
        } // TODO: remove panics!

        let start = *range.start();
        let end = *range.end();

        if start > end || end >= self.shape[dim] {
            panic!("Invalid slice range for dimension {}", dim);
        }

        let mut new_shape = self.shape.clone();
        new_shape[dim] = end - start +1;

        let additional_offset =  start * self.strides[dim];
        
        TensorShape {
            shape: new_shape,
            strides: self.strides.clone(),
            linear_offset: self.linear_offset + additional_offset
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// A struct that stores the data of a tensor as a linear vector.
pub struct TensorStorage<T> {
    data: Vec<T>,
}

impl<T> Index<usize> for TensorStorage<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for TensorStorage<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: Zero + Clone> TensorStorage<T> {
    /// Creates a new TensorStorage filled with zeroes, give the total number of elements.
    fn zeroes(size: usize) -> Self {
        let data: Vec<T> = vec![T::zero(); size];
        TensorStorage { data }
    }
}

impl<T: One + Clone> TensorStorage<T> {
    /// Creates a new TensorStorage filled with ones, give the total number of elements.
    fn ones(size: usize) -> Self {
        let data: Vec<T> = vec![T::one(); size];
        TensorStorage { data }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// A struct representing a multi-dimensional array (tensor).
/// The shape od the tensor is stored in a TensorShaper struct whereas the data is stored
/// in a TensorStorage struct as a liner vector (row-major order).
///
/// # Example
/// ```rust
/// use rustcv::core::Tensor;
/// let tensor = Tensor::<f32>::zeroes(vec![2, 3]);
/// println!("Tensor: {:?}", tensor);
/// ```
pub struct Tensor<T> {
    shape: TensorShape,
    storage: TensorStorage<T>,
}

impl<T> Index<&[usize]> for Tensor<T> {
    type Output = T;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        &self.storage[self.shape.ravel_index(indices)]
    }
}

impl<T> IndexMut<&[usize]> for Tensor<T> {
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        &mut self.storage[self.shape.ravel_index(indices)]
    }
}

impl<T: Zero + Clone> Tensor<T> {
    /// Creates a new Tensor filled with zeroes, give the total tensor's shape.
    pub fn zeroes(shape: Vec<usize>) -> Self {
        let shape: TensorShape = TensorShape::new(shape);
        let storage: TensorStorage<T> = TensorStorage::<T>::zeroes(shape.size());
        Tensor { shape, storage }
    }
}

impl<T: One + Clone> Tensor<T> {
    /// Creates a new Tensor filled with ones, give the total tensor's shape.
    pub fn ones(shape: Vec<usize>) -> Self {
        let shape: TensorShape = TensorShape::new(shape);
        let storage: TensorStorage<T> = TensorStorage::<T>::ones(shape.size());
        Tensor { shape, storage }
    }
}

impl<T: Clone> Tensor<T> {
    /// Permutes the dimensions of the tensor according to the given order of indices.
    pub fn permute(&self, permuted_indices: &[usize]) -> Self {
        Tensor {
            shape: self.shape.permute(permuted_indices),
            storage: self.storage.clone(), // Not efficient, need to implement storage as Arc
        }
    }

    // Merges consecutive dimensions of the tensor.
    pub fn merge(&self, to_merge: RangeInclusive<usize>) -> Self {
        Tensor {
            shape: self.shape.merge(to_merge),
            storage: self.storage.clone(), // Not efficient, need to implement storage as Arc
        }
    }

    // Splits one dimension into a higher number of complatible dimensions.
    pub fn split(&self, dim: usize, shape: &[usize]) -> Self {
        Tensor {
            shape: self.shape.split(dim, shape),
            storage: self.storage.clone(), // Not efficient, need to implement storage as Arc
        }
    }

    pub fn reshape(&self, _shape: &[usize]) -> Self {
        unimplemented!();
    }

    fn slice(&self, dim: usize, range: RangeInclusive<usize>) -> Self {
        Tensor {
            shape: self.shape.slice(dim, range),
            storage: self.storage.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    // importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_zeroes() {
        let tensor: Tensor<f32> = Tensor::<f32>::zeroes(vec![3, 3, 2]);
        assert_eq!(tensor.shape.shape, vec![3, 3, 2]);
        assert_eq!(tensor.storage.data.len(), 18);
        assert!(tensor.storage.data.iter().all(|x| *x == 0.0))
    }

    #[test]
    fn test_ones() {
        let tensor: Tensor<i32> = Tensor::<i32>::ones(vec![1, 5, 5, 3, 2]);
        assert_eq!(tensor.shape.shape, vec![1, 5, 5, 3, 2]);
        assert_eq!(tensor.storage.data.len(), 150);
        assert!(tensor.storage.data.iter().all(|x| *x == 1))
    }

    #[test]
    fn test_ravel_index() {
        let shape: TensorShape = TensorShape::new(vec![4]);
        assert_eq!(shape.ravel_index(&[0]), 0);
        assert_eq!(shape.ravel_index(&[1]), 1);
        assert_eq!(shape.ravel_index(&[2]), 2);
        assert_eq!(shape.ravel_index(&[3]), 3);

        let shape: TensorShape = TensorShape::new(vec![3, 2]);
        assert_eq!(shape.ravel_index(&[0, 0]), 0);
        assert_eq!(shape.ravel_index(&[0, 1]), 1);
        assert_eq!(shape.ravel_index(&[1, 0]), 2);
        assert_eq!(shape.ravel_index(&[1, 1]), 3);
        assert_eq!(shape.ravel_index(&[2, 0]), 4);
        assert_eq!(shape.ravel_index(&[2, 1]), 5);

        let shape: TensorShape = TensorShape::new(vec![3, 3, 3]);
        assert_eq!(shape.ravel_index(&[0, 0, 0]), 0);
        assert_eq!(shape.ravel_index(&[0, 0, 1]), 1);
        assert_eq!(shape.ravel_index(&[0, 0, 2]), 2);
        assert_eq!(shape.ravel_index(&[0, 1, 0]), 3);
        assert_eq!(shape.ravel_index(&[0, 1, 1]), 4);
        assert_eq!(shape.ravel_index(&[0, 1, 2]), 5);
        assert_eq!(shape.ravel_index(&[0, 2, 0]), 6);
        assert_eq!(shape.ravel_index(&[0, 2, 1]), 7);
        assert_eq!(shape.ravel_index(&[0, 2, 2]), 8);
        assert_eq!(shape.ravel_index(&[1, 0, 0]), 9);
        assert_eq!(shape.ravel_index(&[1, 0, 1]), 10);
        assert_eq!(shape.ravel_index(&[1, 0, 2]), 11);
        assert_eq!(shape.ravel_index(&[1, 1, 0]), 12);
        assert_eq!(shape.ravel_index(&[1, 1, 1]), 13);
        assert_eq!(shape.ravel_index(&[1, 1, 2]), 14);
        assert_eq!(shape.ravel_index(&[1, 2, 0]), 15);
        assert_eq!(shape.ravel_index(&[1, 2, 1]), 16);
        assert_eq!(shape.ravel_index(&[1, 2, 2]), 17);
        assert_eq!(shape.ravel_index(&[2, 0, 0]), 18);
        assert_eq!(shape.ravel_index(&[2, 0, 1]), 19);
        assert_eq!(shape.ravel_index(&[2, 0, 2]), 20);
        assert_eq!(shape.ravel_index(&[2, 1, 0]), 21);
        assert_eq!(shape.ravel_index(&[2, 1, 1]), 22);
        assert_eq!(shape.ravel_index(&[2, 1, 2]), 23);
        assert_eq!(shape.ravel_index(&[2, 2, 0]), 24);
        assert_eq!(shape.ravel_index(&[2, 2, 1]), 25);
        assert_eq!(shape.ravel_index(&[2, 2, 2]), 26);

        let shape: TensorShape = TensorShape::new(vec![1, 1, 1]);
        assert_eq!(shape.ravel_index(&[0, 0, 0]), 0);

        let shape: TensorShape = TensorShape::new(vec![3, 3]);
        let expected = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        let mut idx = 0;
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(shape.ravel_index(&[i, j]), expected[idx]);
                idx += 1;
            }
        }
    }

    #[test]
    fn test_unravel_index() {
        let shape: TensorShape = TensorShape::new(vec![]);
        assert_eq!(shape.unravel_index(0), vec![]);

        let shape: TensorShape = TensorShape::new(vec![4]);
        assert_eq!(shape.unravel_index(0), vec![0]);
        assert_eq!(shape.unravel_index(1), vec![1]);
        assert_eq!(shape.unravel_index(2), vec![2]);
        assert_eq!(shape.unravel_index(3), vec![3]);

        let shape: TensorShape = TensorShape::new(vec![3, 2]);
        assert_eq!(shape.unravel_index(0), vec![0, 0]);
        assert_eq!(shape.unravel_index(1), vec![0, 1]);
        assert_eq!(shape.unravel_index(2), vec![1, 0]);
        assert_eq!(shape.unravel_index(3), vec![1, 1]);
        assert_eq!(shape.unravel_index(4), vec![2, 0]);
        assert_eq!(shape.unravel_index(5), vec![2, 1]);

        let shape: TensorShape = TensorShape::new(vec![2, 2, 2]);
        assert_eq!(shape.unravel_index(0), vec![0, 0, 0]);
        assert_eq!(shape.unravel_index(1), vec![0, 0, 1]);
        assert_eq!(shape.unravel_index(2), vec![0, 1, 0]);
        assert_eq!(shape.unravel_index(3), vec![0, 1, 1]);
        assert_eq!(shape.unravel_index(4), vec![1, 0, 0]);
        assert_eq!(shape.unravel_index(5), vec![1, 0, 1]);
        assert_eq!(shape.unravel_index(6), vec![1, 1, 0]);
        assert_eq!(shape.unravel_index(7), vec![1, 1, 1]);

        let shape: TensorShape = TensorShape::new(vec![1, 1, 1, 1]);
        assert_eq!(shape.unravel_index(0), vec![0, 0, 0, 0]);

        let shape: TensorShape = TensorShape::new(vec![3, 4]);

        let expected_indices: [Vec<usize>; 12] = [
            vec![0, 0],
            vec![0, 1],
            vec![0, 2],
            vec![0, 3],
            vec![1, 0],
            vec![1, 1],
            vec![1, 2],
            vec![1, 3],
            vec![2, 0],
            vec![2, 1],
            vec![2, 2],
            vec![2, 3],
        ];

        for (i, expected) in expected_indices.iter().enumerate() {
            assert_eq!(shape.unravel_index(i), *expected)
        }
    }

    #[test]
    fn test_tensor_storage_and_consistency() {
        let mut storage: TensorStorage<u8> = TensorStorage::<u8>::zeroes(6);
        assert!(storage.data.iter().all(|x| *x == 0));

        storage[0] = 100;
        storage[2] = 200;
        storage[4] = 255;
        storage[5] = 2;

        assert_eq!(storage[0], 100);
        assert_eq!(storage[1], 0);
        assert_eq!(storage[2], 200);
        assert_eq!(storage[3], 0);
        assert_eq!(storage[4], 255);
        assert_eq!(storage[5], 2);
    }

    #[test]
    fn test_reshape() {
        let tensor: Tensor<i32> = Tensor::<i32>::ones(vec![2, 3]);
        let permuted_tensor = tensor.permute(&[1, 0]);
        assert_eq!(permuted_tensor.shape.shape, vec![3, 2]);
        assert_eq!(permuted_tensor.storage.data.len(), 6);
        assert!(permuted_tensor.storage.data.iter().all(|x| *x == 1));

        let mut tensor: Tensor<i32> = Tensor::<i32>::ones(vec![2, 3]);
        tensor[&[0, 0]] = 0;
        tensor[&[1, 1]] = 0;
        let permuted_tensor = tensor.permute(&[1, 0]);
        assert_eq!(permuted_tensor.shape.shape, vec![3, 2]);
        assert_eq!(permuted_tensor.storage.data.len(), 6);
        assert_eq!(permuted_tensor[&[0, 0]], 0);
        assert_eq!(permuted_tensor[&[1, 1]], 0);
        assert_eq!(permuted_tensor.storage.data.iter().sum::<i32>(), 4);

        let mut tensor: Tensor<i32> = Tensor::<i32>::zeroes(vec![2, 3]);
        tensor[&[0, 2]] = 10;
        tensor[&[0, 1]] = 8;
        let permuted_tensor = tensor.permute(&[1, 0]);
        assert_eq!(permuted_tensor.shape.shape, vec![3, 2]);
        assert_eq!(permuted_tensor.storage.data.len(), 6);
        assert_eq!(permuted_tensor[&[2, 0]], 10);
        assert_eq!(permuted_tensor[&[1, 0]], 8);
        assert_eq!(permuted_tensor.storage.data.iter().sum::<i32>(), 18);

        let mut tensor: Tensor<i32> = Tensor::<i32>::zeroes(vec![2, 3, 4]);
        tensor[&[0, 2, 1]] = 10;
        tensor[&[0, 1, 3]] = 8;
        let permuted_tensor = tensor.permute(&[1, 2, 0]);
        assert_eq!(permuted_tensor.shape.shape, vec![3, 4, 2]);
        assert_eq!(permuted_tensor.storage.data.len(), 24);
        assert_eq!(permuted_tensor[&[2, 1, 0]], 10);
        assert_eq!(permuted_tensor[&[1, 3, 0]], 8);
        assert_eq!(permuted_tensor.storage.data.iter().sum::<i32>(), 18);
    }

    #[test]
    fn test_merge() {
        let shape = TensorShape::new(vec![2, 3, 4, 5]);
        assert_eq!(shape.shape, vec![2, 3, 4, 5]);
        assert_eq!(shape.strides, Vec::<usize>::from_iter([60, 20, 5, 1]));

        let merged_shape = shape.merge(1..=3);
        assert_eq!(merged_shape.shape, vec![2, 60]);
        assert_eq!(merged_shape.strides, Vec::<usize>::from_iter([60, 1]));

        let merged_shape = shape.merge(1..=2);
        assert_eq!(merged_shape.shape, vec![2, 12, 5]);
        assert_eq!(merged_shape.strides, Vec::<usize>::from_iter([60, 5, 1]));

        let merged_shape = shape.merge(0..=2);
        assert_eq!(merged_shape.shape, vec![24, 5]);
        assert_eq!(merged_shape.strides, Vec::<usize>::from_iter([5, 1]));

        let merged_shape = shape.merge(0..=3);
        assert_eq!(merged_shape.shape, vec![120]);
        assert_eq!(merged_shape.strides, Vec::<usize>::from_iter([1]));
    }

    #[test]
    fn test_split() {
        let shape = TensorShape::new(vec![2, 12, 5]);
        let split_shape = shape.split(1, &[3, 4]);
        assert_eq!(split_shape.shape, vec![2, 3, 4, 5]);
        assert_eq!(split_shape.strides, vec![60, 20, 5, 1]);

        let shape = TensorShape::new(vec![24]);
        let split_shape = shape.split(0, &[2, 3, 0]);
        assert_eq!(split_shape.shape, vec![2, 3, 4]);
        assert_eq!(split_shape.strides, vec![12, 4, 1]);

        let shape = TensorShape::new(vec![2, 3, 24]);
        let split_shape = shape.split(2, &[4, 6]);
        assert_eq!(split_shape.shape, vec![2, 3, 4, 6]);
        assert_eq!(split_shape.strides, vec![72, 24, 6, 1]);

        let shape = TensorShape::new(vec![12, 3, 4]);
        let split_shape = shape.split(0, &[3, 4]);
        assert_eq!(split_shape.shape, vec![3, 4, 3, 4]);
        assert_eq!(split_shape.strides, vec![48, 12, 4, 1]);

        let shape = TensorShape::new(vec![30]);
        let split_shape = shape.split(0, &[5, 6]);
        assert_eq!(split_shape.shape, vec![5, 6]);
        assert_eq!(split_shape.strides, vec![6, 1]);

        let shape = TensorShape::new(vec![2, 60, 3]);
        let split_shape = shape.split(1, &[4, 0, 5]);
        assert_eq!(split_shape.shape, vec![2, 4, 3, 5, 3]);
        assert_eq!(split_shape.strides, vec![180, 45, 15, 3, 1]);

        let original_shape = TensorShape::new(vec![6, 8]);
        let split_shape = original_shape.split(0, &[2, 3]).split(2, &[4, 2]);

        assert_eq!(
            original_shape.ravel_index(&[0, 0]),
            split_shape.ravel_index(&[0, 0, 0, 0])
        );
        assert_eq!(
            original_shape.ravel_index(&[2, 6]),
            split_shape.ravel_index(&[0, 2, 3, 0])
        );
        assert_eq!(
            original_shape.ravel_index(&[5, 7]),
            split_shape.ravel_index(&[1, 2, 3, 1])
        );

        let shape = TensorShape::new(vec![4]);
        let split_shape = shape.split(0, &[4, 1]);
        assert_eq!(split_shape.shape, vec![4, 1]);
        assert_eq!(split_shape.strides, vec![1, 1]);

        let shape = TensorShape::new(vec![2, 3, 4]);
        let split_shape = shape.split(1, &[1, 3]);
        assert_eq!(split_shape.shape, vec![2, 1, 3, 4]);
        assert_eq!(split_shape.strides, vec![12, 12, 4, 1]);
    }

    #[test]
    fn test_slice() {
        let shape = TensorShape::new(vec![2, 6, 3]);
        let sliced_shape = shape.slice(1, 1..=3);
        assert_eq!(sliced_shape.shape, vec![2, 3, 3]);
        assert_eq!(sliced_shape.strides, shape.strides);
        assert_eq!(sliced_shape.linear_offset, 1)
    }
}
