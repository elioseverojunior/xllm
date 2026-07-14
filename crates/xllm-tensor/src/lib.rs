// SPDX-FileCopyrightText: 2026 XLLM contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#[allow(unnameable_types)]
mod sealed {
    pub trait Sealed {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
}

impl DType {
    #[must_use]
    pub const fn byte_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("index {index:?} out of bounds for shape {shape:?}")]
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
    #[error("incompatible dtypes: {a:?} and {b:?}")]
    DTypeMismatch { a: DType, b: DType },
    #[error("element count mismatch: {expected} != {actual}")]
    ElementCount { expected: usize, actual: usize },
    #[error("invalid operation: {0}")]
    InvalidOperation(String),
}

pub type Result<T> = std::result::Result<T, TensorError>;

impl sealed::Sealed for f32 {}
impl sealed::Sealed for half::f16 {}

pub trait DTypeValue: sealed::Sealed + Copy + Default {
    fn dtype() -> DType;
    fn byte_size() -> usize;
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
    fn to_ne_bytes(self) -> [u8; 4];
    fn from_ne_bytes(bytes: [u8; 4]) -> Self;
}

impl DTypeValue for f32 {
    fn dtype() -> DType {
        DType::F32
    }

    fn byte_size() -> usize {
        4
    }

    fn to_f32(self) -> f32 {
        self
    }

    fn from_f32(v: f32) -> Self {
        v
    }

    fn to_ne_bytes(self) -> [u8; 4] {
        Self::to_ne_bytes(self)
    }

    fn from_ne_bytes(bytes: [u8; 4]) -> Self {
        Self::from_ne_bytes(bytes)
    }
}

impl DTypeValue for half::f16 {
    fn dtype() -> DType {
        DType::F16
    }

    fn byte_size() -> usize {
        2
    }

    fn to_f32(self) -> f32 {
        Self::to_f32(self)
    }

    fn from_f32(v: f32) -> Self {
        Self::from_f32(v)
    }

    fn to_ne_bytes(self) -> [u8; 4] {
        let mut bytes = [0u8; 4];
        let short = Self::to_ne_bytes(self);
        bytes[..2].copy_from_slice(&short);
        bytes
    }

    fn from_ne_bytes(bytes: [u8; 4]) -> Self {
        let mut short = [0u8; 2];
        short.copy_from_slice(&bytes[..2]);
        Self::from_ne_bytes(short)
    }
}

fn compute_strides(shape: &[usize], elem_size: usize) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![0usize; ndim];
    if ndim > 0 {
        strides[ndim - 1] = elem_size;
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    strides
}

fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);
    for i in 0..max_ndim {
        let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
            return Err(TensorError::InvalidOperation(format!(
                "cannot broadcast shapes {a:?} and {b:?}"
            )));
        }
        result.push(a_dim.max(b_dim));
    }
    result.reverse();
    Ok(result)
}

fn unflatten_index(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0usize; shape.len()];
    for d in (0..shape.len()).rev() {
        indices[d] = idx % shape[d];
        idx /= shape[d];
    }
    indices
}

fn broadcast_map(result_indices: &[usize], source_shape: &[usize]) -> Vec<usize> {
    let offset = result_indices.len() - source_shape.len();
    source_shape
        .iter()
        .enumerate()
        .map(|(i, &dim)| {
            let ri = result_indices[offset + i];
            if dim == 1 { 0 } else { ri }
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<u8>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    dtype: DType,
}

impl Tensor {
    /// Creates a tensor from a slice of typed values.
    ///
    /// # Errors
    ///
    /// Returns `ElementCount` if `data.len()` does not match the product of `shape`.
    pub fn from_slice<T: DTypeValue>(data: &[T], shape: &[usize]) -> Result<Self> {
        let elem_count: usize = shape.iter().product();
        if data.len() != elem_count {
            return Err(TensorError::ElementCount {
                expected: elem_count,
                actual: data.len(),
            });
        }
        let strides = compute_strides(shape, T::byte_size());
        let byte_count = elem_count * T::byte_size();
        let mut bytes = Vec::with_capacity(byte_count);
        for val in data {
            bytes.extend_from_slice(&val.to_ne_bytes()[..T::byte_size()]);
        }
        Ok(Self {
            data: bytes,
            shape: shape.to_vec(),
            strides,
            dtype: T::dtype(),
        })
    }

    #[must_use]
    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let elem_count: usize = shape.iter().product();
        let byte_size = elem_count * dtype.byte_size();
        let strides = compute_strides(shape, dtype.byte_size());
        Self {
            data: vec![0u8; byte_size],
            shape: shape.to_vec(),
            strides,
            dtype,
        }
    }

    #[must_use]
    pub fn ones(shape: &[usize], dtype: DType) -> Self {
        let elem_count: usize = shape.iter().product();
        let byte_size = elem_count * dtype.byte_size();
        let strides = compute_strides(shape, dtype.byte_size());
        let ones_bytes = match dtype {
            DType::F32 => f32::to_ne_bytes(1.0).to_vec(),
            DType::F16 => half::f16::from_f32(1.0).to_ne_bytes().to_vec(),
        };
        let mut data = Vec::with_capacity(byte_size);
        for _ in 0..elem_count {
            data.extend_from_slice(&ones_bytes);
        }
        Self {
            data,
            shape: shape.to_vec(),
            strides,
            dtype,
        }
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub const fn dims(&self) -> usize {
        self.shape.len()
    }

    #[must_use]
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let expected = compute_strides(&self.shape, self.dtype.byte_size());
        self.strides == expected
    }

    /// Reads an element at the given indices.
    ///
    /// # Errors
    ///
    /// Returns `DTypeMismatch` if `T` does not match the tensor's dtype.
    /// Returns `IndexOutOfBounds` if any index exceeds the corresponding dimension.
    pub fn get<T: DTypeValue>(&self, indices: &[usize]) -> Result<T> {
        if T::dtype() != self.dtype {
            return Err(TensorError::DTypeMismatch {
                a: T::dtype(),
                b: self.dtype,
            });
        }
        let offset = self.byte_offset(indices)?;
        let esize = self.dtype.byte_size();
        let mut buf = [0u8; 4];
        buf[..esize].copy_from_slice(&self.data[offset..offset + esize]);
        Ok(T::from_ne_bytes(buf))
    }

    /// Writes an element at the given indices.
    ///
    /// # Errors
    ///
    /// Returns `DTypeMismatch` if `T` does not match the tensor's dtype.
    /// Returns `IndexOutOfBounds` if any index exceeds the corresponding dimension.
    pub fn set<T: DTypeValue>(&mut self, indices: &[usize], value: T) -> Result<()> {
        if T::dtype() != self.dtype {
            return Err(TensorError::DTypeMismatch {
                a: T::dtype(),
                b: self.dtype,
            });
        }
        let offset = self.byte_offset(indices)?;
        let esize = self.dtype.byte_size();
        let bytes = value.to_ne_bytes();
        self.data[offset..offset + esize].copy_from_slice(&bytes[..esize]);
        Ok(())
    }

    /// Reshapes the tensor into a new shape (contiguous required).
    ///
    /// # Errors
    ///
    /// Returns `ElementCount` if the new shape's element count differs.
    /// Returns `InvalidOperation` if the tensor is not contiguous.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let new_elem_count: usize = new_shape.iter().product();
        if new_elem_count != self.size() {
            return Err(TensorError::ElementCount {
                expected: self.size(),
                actual: new_elem_count,
            });
        }
        if !self.is_contiguous() {
            return Err(TensorError::InvalidOperation(
                "reshape requires a contiguous tensor".to_string(),
            ));
        }
        let strides = compute_strides(new_shape, self.dtype.byte_size());
        Ok(Self {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            strides,
            dtype: self.dtype,
        })
    }

    /// Returns a strided slice view (no data rearrangement).
    ///
    /// # Errors
    ///
    /// Returns `InvalidOperation` if the number of ranges does not match tensor dimensions.
    /// Returns `IndexOutOfBounds` if any range exceeds the corresponding dimension.
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Result<Self> {
        if ranges.len() != self.shape.len() {
            return Err(TensorError::InvalidOperation(format!(
                "slice dimension mismatch: expected {}, got {}",
                self.shape.len(),
                ranges.len()
            )));
        }
        let mut new_shape = Vec::with_capacity(ranges.len());
        let mut data_offset: usize = 0;
        for (dim, range) in ranges.iter().enumerate() {
            if range.start > range.end || range.end > self.shape[dim] {
                return Err(TensorError::IndexOutOfBounds {
                    index: vec![range.start, range.end],
                    shape: self.shape.clone(),
                });
            }
            data_offset += range.start * self.strides[dim];
            new_shape.push(range.end - range.start);
        }
        let strides = self.strides.clone();
        let data = self.data[data_offset..].to_vec();
        Ok(Self {
            data,
            shape: new_shape,
            strides,
            dtype: self.dtype,
        })
    }

    /// Returns a strided transpose view (no data rearrangement).
    ///
    /// # Errors
    ///
    /// Returns `IndexOutOfBounds` if either dimension is out of range.
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        if dim1 >= self.shape.len() || dim2 >= self.shape.len() {
            return Err(TensorError::IndexOutOfBounds {
                index: vec![dim1, dim2],
                shape: self.shape.clone(),
            });
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(dim1, dim2);
        strides.swap(dim1, dim2);
        Ok(Self {
            data: self.data.clone(),
            shape,
            strides,
            dtype: self.dtype,
        })
    }

    /// Performs matrix multiplication (f32 only, naive O(n³) loop).
    ///
    /// # Panics
    ///
    /// Panics if the byte offsets are invalid — this indicates a logic bug since
    /// all indices are validated before access.
    ///
    /// # Errors
    ///
    /// Returns `DTypeMismatch` if either tensor is not f32.
    /// Returns `InvalidOperation` if either tensor is not 2D.
    /// Returns `ShapeMismatch` if inner dimensions do not match.
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        if self.dtype != DType::F32 || other.dtype != DType::F32 {
            return Err(TensorError::DTypeMismatch {
                a: self.dtype,
                b: other.dtype,
            });
        }
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(TensorError::InvalidOperation(
                "matmul requires 2D tensors".to_string(),
            ));
        }
        let m = self.shape[0];
        let k1 = self.shape[1];
        let k2 = other.shape[0];
        let n = other.shape[1];
        if k1 != k2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![m, k1],
                got: vec![k2, n],
            });
        }
        let mut result = Self::zeros(&[m, n], DType::F32);
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for k in 0..k1 {
                    let a = self.get::<f32>(&[i, k])?;
                    let b = other.get::<f32>(&[k, j])?;
                    sum = a.mul_add(b, sum);
                }
                result.set(&[i, j], sum)?;
            }
        }
        Ok(result)
    }

    /// Element-wise addition with NumPy-style broadcasting.
    ///
    /// # Panics
    ///
    /// Panics if the byte offsets are invalid — this indicates a logic bug since
    /// all indices are validated before access.
    ///
    /// # Errors
    ///
    /// Returns `DTypeMismatch` if the tensors have different dtypes.
    /// Returns `InvalidOperation` if shapes are not broadcastable.
    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.dtype != other.dtype {
            return Err(TensorError::DTypeMismatch {
                a: self.dtype,
                b: other.dtype,
            });
        }
        let result_shape = broadcast_shapes(&self.shape, &other.shape)?;
        let mut result = Self::zeros(&result_shape, self.dtype);
        let total = result.size();

        for flat in 0..total {
            let indices = unflatten_index(flat, &result_shape);
            let a_idx = broadcast_map(&indices, &self.shape);
            let b_idx = broadcast_map(&indices, &other.shape);
            let a_off = self.byte_offset(&a_idx)?;
            let b_off = other.byte_offset(&b_idx)?;
            let r_off = result.byte_offset(&indices)?;

            match self.dtype {
                DType::F32 => {
                    let mut a_buf = [0u8; 4];
                    let mut b_buf = [0u8; 4];
                    a_buf.copy_from_slice(&self.data[a_off..a_off + 4]);
                    b_buf.copy_from_slice(&other.data[b_off..b_off + 4]);
                    let a = f32::from_ne_bytes(a_buf);
                    let b = f32::from_ne_bytes(b_buf);
                    result.data[r_off..r_off + 4].copy_from_slice(&f32::to_ne_bytes(a + b));
                }
                DType::F16 => {
                    let mut a_buf = [0u8; 2];
                    let mut b_buf = [0u8; 2];
                    a_buf.copy_from_slice(&self.data[a_off..a_off + 2]);
                    b_buf.copy_from_slice(&other.data[b_off..b_off + 2]);
                    let a = half::f16::from_ne_bytes(a_buf);
                    let b = half::f16::from_ne_bytes(b_buf);
                    let r = half::f16::from_f32(a.to_f32() + b.to_f32());
                    result.data[r_off..r_off + 2].copy_from_slice(&r.to_ne_bytes());
                }
            }
        }
        Ok(result)
    }

    #[must_use]
    pub fn to_owned(&self) -> Self {
        let strides = compute_strides(&self.shape, self.dtype.byte_size());
        let elem_count = self.size();
        let esize = self.dtype.byte_size();
        let mut data = vec![0u8; elem_count * esize];

        if self.is_contiguous() {
            data.copy_from_slice(&self.data[..elem_count * esize]);
        } else {
            let mut indices = vec![0usize; self.shape.len()];
            self.materialise_into(&mut data, &mut indices, 0, &mut 0);
        }

        Self {
            data,
            shape: self.shape.clone(),
            strides,
            dtype: self.dtype,
        }
    }

    /// Returns a contiguous copy of the tensor, materialising strided views.
    ///
    /// # Errors
    ///
    /// This operation never fails — the `Result` type is used for future
    /// extensibility.
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            let strides = compute_strides(&self.shape, self.dtype.byte_size());
            return Ok(Self {
                data: self.data.clone(),
                shape: self.shape.clone(),
                strides,
                dtype: self.dtype,
            });
        }
        Ok(self.to_owned())
    }

    fn materialise_into(
        &self,
        dst: &mut [u8],
        indices: &mut [usize],
        dim: usize,
        flat_idx: &mut usize,
    ) {
        let esize = self.dtype.byte_size();
        if dim == indices.len() {
            let mut src_off: usize = 0;
            for (d, &idx) in indices.iter().enumerate() {
                src_off += idx * self.strides[d];
            }
            let dst_off = *flat_idx * esize;
            dst[dst_off..dst_off + esize].copy_from_slice(&self.data[src_off..src_off + esize]);
            *flat_idx += 1;
            return;
        }
        for i in 0..self.shape[dim] {
            indices[dim] = i;
            self.materialise_into(dst, indices, dim + 1, flat_idx);
        }
    }

    fn byte_offset(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.shape.len() {
            return Err(TensorError::IndexOutOfBounds {
                index: indices.to_vec(),
                shape: self.shape.clone(),
            });
        }
        let mut offset: usize = 0;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(TensorError::IndexOutOfBounds {
                    index: indices.to_vec(),
                    shape: self.shape.clone(),
                });
            }
            offset += idx * self.strides[i];
        }
        Ok(offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // 1. Construction and accessors
    // -----------------------------------------------------------------------
    #[test]
    fn test_from_slice_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = Tensor::from_slice(&data, &[2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.dims(), 2);
        assert_eq!(t.size(), 4);
        assert_eq!(t.dtype(), DType::F32);
        assert!(t.is_contiguous());
    }

    #[test]
    fn test_from_slice_f16() {
        let data = vec![half::f16::from_f32(1.0), half::f16::from_f32(2.0)];
        let t = Tensor::from_slice(&data, &[2]).unwrap();
        assert_eq!(t.shape(), &[2]);
        assert_eq!(t.size(), 2);
        assert_eq!(t.dtype(), DType::F16);
        assert!(t.is_contiguous());
    }

    #[test]
    fn test_from_slice_1d() {
        let data = vec![10.0f32, 20.0, 30.0];
        let t = Tensor::from_slice(&data, &[3]).unwrap();
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.dims(), 1);
        assert_eq!(t.size(), 3);
    }

    #[test]
    fn test_from_slice_scalar() {
        let data = vec![42.0f32];
        let t = Tensor::from_slice(&data, &[1]).unwrap();
        assert_eq!(t.shape(), &[1]);
        assert_eq!(t.size(), 1);
    }

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(&[2, 3], DType::F32);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.size(), 6);
        assert_eq!(t.dtype(), DType::F32);
        assert!(t.is_contiguous());
        for i in 0..2 {
            for j in 0..3 {
                assert!((t.get::<f32>(&[i, j]).unwrap() - 0.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_zeros_f16() {
        let t = Tensor::zeros(&[4], DType::F16);
        for i in 0..4 {
            let val: half::f16 = t.get(&[i]).unwrap();
            assert!((val.to_f32() - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ones() {
        let t = Tensor::ones(&[2, 2], DType::F32);
        for i in 0..2 {
            for j in 0..2 {
                assert!((t.get::<f32>(&[i, j]).unwrap() - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_ones_f16() {
        let t = Tensor::ones(&[3], DType::F16);
        for i in 0..3 {
            let val: half::f16 = t.get(&[i]).unwrap();
            assert!((val.to_f32() - 1.0).abs() < 1e-6);
        }
    }

    // -----------------------------------------------------------------------
    // 2. Element get/set
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_set_f32() {
        let mut t = Tensor::zeros(&[2, 3], DType::F32);
        t.set(&[0, 0], 1.5f32).unwrap();
        t.set(&[1, 2], -3.0f32).unwrap();
        assert!((t.get::<f32>(&[0, 0]).unwrap() - 1.5).abs() < 1e-6);
        assert!((t.get::<f32>(&[1, 2]).unwrap() - (-3.0)).abs() < 1e-6);
        assert!((t.get::<f32>(&[0, 1]).unwrap() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_get_set_f16() {
        let mut t = Tensor::zeros(&[2], DType::F16);
        let v1 = half::f16::from_f32(3.25);
        let v2 = half::f16::from_f32(-0.5);
        t.set(&[0], v1).unwrap();
        t.set(&[1], v2).unwrap();
        assert!((t.get::<half::f16>(&[0]).unwrap().to_f32() - 3.25).abs() < 1e-3);
        assert!((t.get::<half::f16>(&[1]).unwrap().to_f32() - (-0.5)).abs() < 1e-3);
    }

    #[test]
    fn test_get_out_of_bounds() {
        let t = Tensor::zeros(&[3, 4], DType::F32);
        assert!(t.get::<f32>(&[3, 0]).is_err());
        assert!(t.get::<f32>(&[0, 4]).is_err());
        assert!(t.get::<f32>(&[0]).is_err());
        assert!(t.get::<f32>(&[0, 0, 0]).is_err());
    }

    #[test]
    fn test_set_out_of_bounds() {
        let mut t = Tensor::zeros(&[2, 2], DType::F32);
        assert!(t.set(&[2, 0], 1.0f32).is_err());
        assert!(t.set(&[0, 2], 1.0f32).is_err());
    }

    // -----------------------------------------------------------------------
    // 3. Shape mismatch errors
    // -----------------------------------------------------------------------
    #[test]
    fn test_from_slice_element_count_mismatch() {
        let data = vec![1.0f32, 2.0, 3.0];
        let err = Tensor::from_slice::<f32>(&data, &[2, 2]).unwrap_err();
        assert!(matches!(err, TensorError::ElementCount { .. }));
    }

    #[test]
    fn test_dtype_mismatch_get() {
        let t = Tensor::zeros(&[3], DType::F32);
        let err = t.get::<half::f16>(&[0]).unwrap_err();
        assert!(matches!(err, TensorError::DTypeMismatch { .. }));
    }

    #[test]
    fn test_dtype_mismatch_set() {
        let mut t = Tensor::zeros(&[3], DType::F16);
        let err = t.set(&[0], 1.0f32).unwrap_err();
        assert!(matches!(err, TensorError::DTypeMismatch { .. }));
    }

    // -----------------------------------------------------------------------
    // 4. Reshape
    // -----------------------------------------------------------------------
    #[test]
    fn test_reshape_valid() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_slice(&data, &[2, 3]).unwrap();
        let r = t.reshape(&[3, 2]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.size(), 6);
        assert!(r.is_contiguous());
        assert!((r.get::<f32>(&[0, 0]).unwrap() - 1.0).abs() < 1e-6);
        assert!((r.get::<f32>(&[2, 1]).unwrap() - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_reshape_element_count_mismatch() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = Tensor::from_slice(&data, &[2, 2]).unwrap();
        let err = t.reshape(&[3, 2]).unwrap_err();
        assert!(matches!(err, TensorError::ElementCount { .. }));
    }

    #[test]
    fn test_reshape_non_contiguous() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_slice(&data, &[2, 3]).unwrap();
        let transposed = t.transpose(0, 1).unwrap();
        assert!(!transposed.is_contiguous());
        let err = transposed.reshape(&[3, 2]).unwrap_err();
        assert!(matches!(err, TensorError::InvalidOperation(_)));
    }

    #[test]
    fn test_reshape_1d() {
        let t = Tensor::ones(&[12], DType::F32);
        let r = t.reshape(&[3, 4]).unwrap();
        assert_eq!(r.shape(), &[3, 4]);
    }

    // -----------------------------------------------------------------------
    // 5. Slice
    // -----------------------------------------------------------------------
    #[test]
    fn test_slice_2d() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_slice(&data, &[2, 3]).unwrap();
        let s = t.slice(&[0..2, 0..2]).unwrap();
        assert_eq!(s.shape(), &[2, 2]);
        assert!((s.get::<f32>(&[0, 0]).unwrap() - 1.0).abs() < 1e-6);
        assert!((s.get::<f32>(&[0, 1]).unwrap() - 2.0).abs() < 1e-6);
        assert!((s.get::<f32>(&[1, 0]).unwrap() - 4.0).abs() < 1e-6);
        assert!((s.get::<f32>(&[1, 1]).unwrap() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_slice_single_row() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_slice(&data, &[2, 3]).unwrap();
        let s = t.slice(&[1..2, 0..3]).unwrap();
        assert_eq!(s.shape(), &[1, 3]);
        assert!((s.get::<f32>(&[0, 0]).unwrap() - 4.0).abs() < 1e-6);
        assert!((s.get::<f32>(&[0, 2]).unwrap() - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_slice_invalid_range() {
        let t = Tensor::zeros(&[3, 4], DType::F32);
        assert!(t.slice(&[0..5, 0..2]).is_err());
        assert!(t.slice(&[0..1, 0..2, 0..2]).is_err());
    }

    // -----------------------------------------------------------------------
    // 6. Transpose
    // -----------------------------------------------------------------------
    #[test]
    fn test_transpose_2d() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_slice(&data, &[2, 3]).unwrap();
        let transposed = t.transpose(0, 1).unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        assert!(!transposed.is_contiguous());
        assert!((transposed.get::<f32>(&[0, 0]).unwrap() - 1.0).abs() < 1e-6);
        assert!((transposed.get::<f32>(&[1, 0]).unwrap() - 2.0).abs() < 1e-6);
        assert!((transposed.get::<f32>(&[0, 1]).unwrap() - 4.0).abs() < 1e-6);
        assert!((transposed.get::<f32>(&[2, 1]).unwrap() - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_transpose_invalid_dim() {
        let t = Tensor::zeros(&[2, 3], DType::F32);
        assert!(t.transpose(0, 2).is_err());
        assert!(t.transpose(2, 1).is_err());
    }

    #[test]
    fn test_transpose_identity() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = Tensor::from_slice(&data, &[2, 2]).unwrap();
        let transposed = t.transpose(0, 1).unwrap();
        assert!(!transposed.is_contiguous());
        assert!((transposed.get::<f32>(&[0, 1]).unwrap() - 3.0).abs() < 1e-6);
        assert!((transposed.get::<f32>(&[1, 0]).unwrap() - 2.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // 7. Contiguous materialisation
    // -----------------------------------------------------------------------
    #[test]
    fn test_contiguous_transposed() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_slice(&data, &[2, 3]).unwrap();
        let transposed = t.transpose(0, 1).unwrap();
        assert!(!transposed.is_contiguous());
        let c = transposed.contiguous().unwrap();
        assert!(c.is_contiguous());
        assert_eq!(c.shape(), &[3, 2]);
        assert!((c.get::<f32>(&[0, 0]).unwrap() - 1.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[1, 0]).unwrap() - 2.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[0, 1]).unwrap() - 4.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[2, 1]).unwrap() - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_contiguous_already_contiguous() {
        let t = Tensor::ones(&[3, 4], DType::F32);
        let c = t.contiguous().unwrap();
        assert!(c.is_contiguous());
        assert_eq!(c.shape(), &[3, 4]);
    }

    #[test]
    fn test_contiguous_slice() {
        let data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let t = Tensor::from_slice(&data, &[2, 3]).unwrap();
        let s = t.slice(&[0..2, 1..3]).unwrap();
        assert_eq!(s.shape(), &[2, 2]);
        let c = s.contiguous().unwrap();
        assert!(c.is_contiguous());
        assert!((c.get::<f32>(&[0, 0]).unwrap() - 1.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[0, 1]).unwrap() - 2.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[1, 0]).unwrap() - 4.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[1, 1]).unwrap() - 5.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // 8. MatMul
    // -----------------------------------------------------------------------
    #[test]
    fn test_matmul_2x2() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        // C = A * B
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(c.shape(), &[2, 2]);
        assert!((c.get::<f32>(&[0, 0]).unwrap() - 19.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[0, 1]).unwrap() - 22.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[1, 0]).unwrap() - 43.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[1, 1]).unwrap() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_2x3_3x2() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::from_slice(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert!((c.get::<f32>(&[0, 0]).unwrap() - 58.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[0, 1]).unwrap() - 64.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[1, 0]).unwrap() - 139.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[1, 1]).unwrap() - 154.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_identity() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let i = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let c = a.matmul(&i).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (c.get::<f32>(&[i, j]).unwrap() - a.get::<f32>(&[i, j]).unwrap()).abs() < 1e-6
                );
            }
        }
    }

    #[test]
    fn test_matmul_dtype_mismatch() {
        let a = Tensor::zeros(&[2, 3], DType::F32);
        let b = Tensor::zeros(&[3, 2], DType::F16);
        let err = a.matmul(&b).unwrap_err();
        assert!(matches!(err, TensorError::DTypeMismatch { .. }));
    }

    #[test]
    fn test_matmul_not_2d() {
        let a = Tensor::zeros(&[4], DType::F32);
        let b = Tensor::zeros(&[4], DType::F32);
        let err = a.matmul(&b).unwrap_err();
        assert!(matches!(err, TensorError::InvalidOperation(_)));
    }

    #[test]
    fn test_matmul_inner_dim_mismatch() {
        let a = Tensor::zeros(&[2, 3], DType::F32);
        let b = Tensor::zeros(&[4, 5], DType::F32);
        let err = a.matmul(&b).unwrap_err();
        assert!(matches!(err, TensorError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_matmul_with_strided_tensors() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let a = Tensor::from_slice(&data, &[2, 2]).unwrap();
        let transposed = a.transpose(0, 1).unwrap();
        let result = transposed.matmul(&a).unwrap();
        // A^T * A where A = [[1,2],[3,4]]
        // A^T = [[1,3],[2,4]]
        // A^T * A = [[1*1+3*3, 1*2+3*4], [2*1+4*3, 2*2+4*4]]
        //         = [[10, 14], [14, 20]]
        assert_eq!(result.shape(), &[2, 2]);
        assert!((result.get::<f32>(&[0, 0]).unwrap() - 10.0).abs() < 1e-6);
        assert!((result.get::<f32>(&[0, 1]).unwrap() - 14.0).abs() < 1e-6);
        assert!((result.get::<f32>(&[1, 0]).unwrap() - 14.0).abs() < 1e-6);
        assert!((result.get::<f32>(&[1, 1]).unwrap() - 20.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // 9. Add
    // -----------------------------------------------------------------------
    #[test]
    fn test_add_same_shape() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert!((c.get::<f32>(&[0, 0]).unwrap() - 6.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[1, 1]).unwrap() - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_f16() {
        let a = Tensor::from_slice(&[half::f16::from_f32(1.0), half::f16::from_f32(2.0)], &[2])
            .unwrap();
        let b = Tensor::from_slice(&[half::f16::from_f32(3.0), half::f16::from_f32(4.0)], &[2])
            .unwrap();
        let c = a.add(&b).unwrap();
        let v0: half::f16 = c.get(&[0]).unwrap();
        let v1: half::f16 = c.get(&[1]).unwrap();
        assert!((v0.to_f32() - 4.0).abs() < 1e-3);
        assert!((v1.to_f32() - 6.0).abs() < 1e-3);
    }

    #[test]
    fn test_add_broadcast_scalar() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_slice(&[10.0f32], &[1]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert!((c.get::<f32>(&[0, 0]).unwrap() - 11.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[1, 1]).unwrap() - 14.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_broadcast_row() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::from_slice(&[10.0f32, 20.0, 30.0], &[3]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert!((c.get::<f32>(&[0, 0]).unwrap() - 11.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[0, 1]).unwrap() - 22.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[0, 2]).unwrap() - 33.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[1, 0]).unwrap() - 14.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[1, 1]).unwrap() - 25.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[1, 2]).unwrap() - 36.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_broadcast_column() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_slice(&[10.0f32, 100.0], &[2, 1]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert!((c.get::<f32>(&[0, 0]).unwrap() - 11.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[0, 1]).unwrap() - 12.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[1, 0]).unwrap() - 103.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[1, 1]).unwrap() - 104.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_dtype_mismatch() {
        let a = Tensor::zeros(&[3], DType::F32);
        let b = Tensor::zeros(&[3], DType::F16);
        let err = a.add(&b).unwrap_err();
        assert!(matches!(err, TensorError::DTypeMismatch { .. }));
    }

    #[test]
    fn test_add_incompatible_shapes() {
        let a = Tensor::zeros(&[2, 3], DType::F32);
        let b = Tensor::zeros(&[4, 5], DType::F32);
        let err = a.add(&b).unwrap_err();
        assert!(matches!(err, TensorError::InvalidOperation(_)));
    }

    // -----------------------------------------------------------------------
    // 10. to_owned
    // -----------------------------------------------------------------------
    #[test]
    fn test_to_owned_contiguous() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = Tensor::from_slice(&data, &[2, 2]).unwrap();
        let o = t.to_owned();
        assert!(o.is_contiguous());
        assert_eq!(o.shape(), &[2, 2]);
        assert!((o.get::<f32>(&[0, 0]).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_to_owned_strided() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_slice(&data, &[2, 3]).unwrap();
        let transposed = t.transpose(0, 1).unwrap();
        let o = transposed.to_owned();
        assert!(o.is_contiguous());
        assert_eq!(o.shape(), &[3, 2]);
        assert!((o.get::<f32>(&[0, 0]).unwrap() - 1.0).abs() < 1e-6);
        assert!((o.get::<f32>(&[1, 0]).unwrap() - 2.0).abs() < 1e-6);
        assert!((o.get::<f32>(&[2, 0]).unwrap() - 3.0).abs() < 1e-6);
        assert!((o.get::<f32>(&[0, 1]).unwrap() - 4.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // 11. DType conversions
    // -----------------------------------------------------------------------
    #[test]
    fn test_f32_to_f16_roundtrip() {
        let val = std::f32::consts::PI;
        let f16 = half::f16::from_f32(val);
        let back = f16.to_f32();
        assert!((back - val).abs() < 1e-3);
    }

    #[test]
    fn test_dtype_byte_size() {
        assert_eq!(DType::F32.byte_size(), 4);
        assert_eq!(DType::F16.byte_size(), 2);
    }

    #[test]
    fn test_trait_dtype_value_f32() {
        assert_eq!(<f32 as DTypeValue>::dtype(), DType::F32);
        assert_eq!(<f32 as DTypeValue>::byte_size(), 4);
        assert!((<f32 as DTypeValue>::from_f32(5.0) - 5.0).abs() < 1e-6);
        assert!((3.0f32.to_f32() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_trait_dtype_value_f16() {
        assert_eq!(<half::f16 as DTypeValue>::dtype(), DType::F16);
        assert_eq!(<half::f16 as DTypeValue>::byte_size(), 2);
        let v = <half::f16 as DTypeValue>::from_f32(2.5);
        assert!((v.to_f32() - 2.5).abs() < 1e-3);
    }

    // -----------------------------------------------------------------------
    // 12. Edge cases
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_tensor() {
        let t = Tensor::zeros(&[0], DType::F32);
        assert_eq!(t.shape(), &[0]);
        assert_eq!(t.size(), 0);
        assert!(t.is_contiguous());
    }

    #[test]
    fn test_1d_tensor_operations() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], &[3]).unwrap();
        let c = a.add(&b).unwrap();
        assert!((c.get::<f32>(&[0]).unwrap() - 5.0).abs() < 1e-6);
        assert!((c.get::<f32>(&[2]).unwrap() - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_high_dimensional_transpose() {
        let data = (0u8..24).map(f32::from).collect::<Vec<_>>();
        let t = Tensor::from_slice(&data, &[2, 3, 4]).unwrap();
        let transposed = t.transpose(0, 2).unwrap();
        assert_eq!(transposed.shape(), &[4, 3, 2]);
        assert!((transposed.get::<f32>(&[0, 0, 0]).unwrap() - 0.0).abs() < 1e-6);
        assert!((transposed.get::<f32>(&[3, 2, 1]).unwrap() - 23.0).abs() < 1e-6);
    }
}
