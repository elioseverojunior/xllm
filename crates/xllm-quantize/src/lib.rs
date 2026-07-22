#![allow(

// SPDX-FileCopyrightText: 2026 XLLM contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
)]

use half::f16;
use xllm_tensor::{Tensor, TensorError};

pub const Q4_0_BLOCK_SIZE: usize = 32;
pub const Q4_0_BLOCK_BYTES: usize = 18;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    Q4_0,
}

#[derive(Debug)]
pub struct QuantizationError;

impl std::fmt::Display for QuantizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "quantization error")
    }
}

impl std::error::Error for QuantizationError {}

#[must_use]
pub const fn bytes_for_q4_0(num_elements: usize) -> usize {
    let num_blocks = num_elements.div_ceil(Q4_0_BLOCK_SIZE);
    num_blocks * Q4_0_BLOCK_BYTES
}

fn flat_index_to_indices(idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0usize; shape.len()];
    let mut remaining = idx;
    for d in (0..shape.len()).rev() {
        indices[d] = remaining % shape[d];
        remaining /= shape[d];
    }
    indices
}

fn tensor_val(tensor: &Tensor, flat_idx: usize) -> f32 {
    let indices = flat_index_to_indices(flat_idx, tensor.shape());
    tensor.get::<f32>(&indices).unwrap_or(0.0)
}

pub struct Quantizer;

impl Quantizer {
    #[must_use]
    pub fn quantize_q4_0(tensor: &Tensor) -> Vec<u8> {
        let shape = tensor.shape().to_vec();
        let num_elements: usize = shape.iter().product();
        let num_blocks = num_elements.div_ceil(Q4_0_BLOCK_SIZE);
        let mut result = vec![0u8; num_blocks * Q4_0_BLOCK_BYTES];

        for block_idx in 0..num_blocks {
            let start = block_idx * Q4_0_BLOCK_SIZE;
            let end = (start + Q4_0_BLOCK_SIZE).min(num_elements);

            let mut max_abs = 0.0f32;
            for i in start..end {
                let val = tensor_val(tensor, i);
                max_abs = max_abs.max(val.abs());
            }

            if max_abs == 0.0 {
                continue;
            }

            let d = max_abs / 7.0f32;
            let d_bits = f16::from_f32(d).to_bits();
            let block_offset = block_idx * Q4_0_BLOCK_BYTES;
            result[block_offset..block_offset + 2].copy_from_slice(&d_bits.to_le_bytes());

            for i in start..end {
                let val = tensor_val(tensor, i);
                let q = ((val / d + 8.0).round() as i32).clamp(0, 15) as u8;
                let offset = i - start;
                let byte_idx = block_offset + 2 + offset / 2;
                if offset & 1 == 0 {
                    result[byte_idx] = (result[byte_idx] & 0xf0) | q;
                } else {
                    result[byte_idx] = (result[byte_idx] & 0x0f) | (q << 4);
                }
            }

            for i in end..start + Q4_0_BLOCK_SIZE {
                let offset = i - start;
                let byte_idx = block_offset + 2 + offset / 2;
                if offset & 1 == 0 {
                    result[byte_idx] = (result[byte_idx] & 0xf0) | 8;
                } else {
                    result[byte_idx] = (result[byte_idx] & 0x0f) | (8 << 4);
                }
            }
        }

        result
    }

    /// Dequantize Q4_0 block-compressed data back into a float32 tensor.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::InvalidOperation`] if `data` is too short for
    /// the given `shape`.
    pub fn dequantize_q4_0(data: &[u8], shape: &[usize]) -> Result<Tensor, TensorError> {
        let num_elements: usize = shape.iter().product();
        let expected_size = bytes_for_q4_0(num_elements);
        if data.len() < expected_size {
            return Err(TensorError::InvalidOperation(format!(
                "Q4_0 data too short: {} bytes, need {}",
                data.len(),
                expected_size
            )));
        }

        let mut values = vec![0.0f32; num_elements];
        let num_blocks = num_elements.div_ceil(Q4_0_BLOCK_SIZE);

        for block_idx in 0..num_blocks {
            let start = block_idx * Q4_0_BLOCK_SIZE;
            let end = (start + Q4_0_BLOCK_SIZE).min(num_elements);
            let block_offset = block_idx * Q4_0_BLOCK_BYTES;

            let mut d_bits = [0u8; 2];
            d_bits.copy_from_slice(&data[block_offset..block_offset + 2]);
            let d = f16::from_bits(u16::from_le_bytes(d_bits)).to_f32();

            for (pos, val_ref) in values[start..end].iter_mut().enumerate() {
                let byte_idx = block_offset + 2 + pos / 2;
                let byte_val = data[byte_idx];
                let nibble = if pos & 1 == 0 {
                    byte_val & 0x0f
                } else {
                    (byte_val >> 4) & 0x0f
                };
                *val_ref = (f32::from(nibble) - 8.0) * d;
            }
        }

        Tensor::from_slice(&values, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, epsilon: f32) {
        assert!(
            (a - b).abs() <= epsilon,
            "expected {a} to be close to {b} (epsilon {epsilon})"
        );
    }

    #[test]
    fn test_q4_0_roundtrip_simple() {
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, -1.0, -2.0, 3.0, -3.0, 4.0];
        let tensor = Tensor::from_slice(&data, &[8]).unwrap();
        let q = Quantizer::quantize_q4_0(&tensor);
        let deq = Quantizer::dequantize_q4_0(&q, &[8]).unwrap();

        for (i, &expected) in data.iter().enumerate() {
            let v: f32 = deq.get(&[i]).unwrap();
            assert_close(v, expected, 0.5);
        }
    }

    #[test]
    fn test_q4_0_roundtrip_2d() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let tensor = Tensor::from_slice(&data, &[8, 8]).unwrap();
        let q = Quantizer::quantize_q4_0(&tensor);
        let deq = Quantizer::dequantize_q4_0(&q, &[8, 8]).unwrap();

        for (i, &expected) in data.iter().enumerate() {
            let row = i / 8;
            let col = i % 8;
            let v: f32 = deq.get(&[row, col]).unwrap();
            assert_close(v, expected, 6.0);
        }
    }

    #[test]
    fn test_q4_0_block_boundary() {
        let data: Vec<f32> = (0..33).map(|i| i as f32).collect();
        let tensor = Tensor::from_slice(&data, &[33]).unwrap();
        let q = Quantizer::quantize_q4_0(&tensor);
        let deq = Quantizer::dequantize_q4_0(&q, &[33]).unwrap();

        assert_eq!(deq.shape(), &[33]);
        for (i, &expected) in data.iter().enumerate() {
            let v: f32 = deq.get(&[i]).unwrap();
            assert_close(v, expected, 5.0);
        }
    }

    #[test]
    fn test_q4_0_all_zeros() {
        let data = vec![0.0f32; 64];
        let tensor = Tensor::from_slice(&data, &[8, 8]).unwrap();
        let q = Quantizer::quantize_q4_0(&tensor);
        let deq = Quantizer::dequantize_q4_0(&q, &[8, 8]).unwrap();

        for i in 0..8 {
            for j in 0..8 {
                let v: f32 = deq.get(&[i, j]).unwrap();
                assert_close(v, 0.0, 1e-6);
            }
        }
    }

    #[test]
    fn test_q4_0_byte_size() {
        assert_eq!(bytes_for_q4_0(1), 18);
        assert_eq!(bytes_for_q4_0(32), 18);
        assert_eq!(bytes_for_q4_0(33), 36);
        assert_eq!(bytes_for_q4_0(64), 36);
        assert_eq!(bytes_for_q4_0(65), 54);
    }

    #[test]
    fn test_q4_0_negative_values() {
        let data: Vec<f32> = vec![-8.0, -5.0, -1.0, 0.0, 1.0, 5.0, 8.0, -3.0];
        let tensor = Tensor::from_slice(&data, &[8]).unwrap();
        let q = Quantizer::quantize_q4_0(&tensor);
        let deq = Quantizer::dequantize_q4_0(&q, &[8]).unwrap();

        for (i, &expected) in data.iter().enumerate() {
            let v: f32 = deq.get(&[i]).unwrap();
            assert_close(v, expected, 1.0);
        }
    }

    #[test]
    fn test_q4_0_short_data_returns_error() {
        let result = Quantizer::dequantize_q4_0(&[0u8; 5], &[64]);
        assert!(result.is_err());
    }

    #[test]
    fn quantizer_exists() {
        let _ = Quantizer;
    }
}
