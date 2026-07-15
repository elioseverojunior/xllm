// SPDX-FileCopyrightText: 2026 XLLM contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use xllm_tensor::{Tensor, TensorError};

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SamplingError {
    #[error("tensor error: {0}")]
    TensorError(#[from] TensorError),
    #[error("logits tensor has wrong shape: expected 1D or 2D, got {0:?}")]
    InvalidShape(Vec<usize>),
    #[error("all logits are -inf or NaN after filtering")]
    NoValidTokens,
    #[error("vocabulary size exceeds u32 range")]
    VocabOverflow,
}

pub type Result<T> = std::result::Result<T, SamplingError>;

pub struct Sampler {
    rng: fastrand::Rng,
}

#[allow(clippy::missing_errors_doc)]
impl Sampler {
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: fastrand::Rng::with_seed(seed),
        }
    }

    pub fn sample(&mut self, logits: &Tensor, params: &SamplingParams) -> Result<u32> {
        let mut logits = extract_logits(logits)?;

        if params.temperature > 0.0 {
            apply_temperature(&mut logits, params.temperature);
        }

        if params.top_k > 0 {
            apply_top_k(&mut logits, params.top_k);
        }

        if params.top_p < 1.0 {
            apply_top_p(&mut logits, params.top_p);
        }

        let probs = softmax(&logits);

        if probs.iter().all(|&p| p.is_nan() || p <= 0.0) {
            return Err(SamplingError::NoValidTokens);
        }

        if params.temperature == 0.0 || params.top_k == 1 {
            return u32::try_from(argmax(&logits)).map_err(|_| SamplingError::VocabOverflow);
        }

        let token = self.sample_multinomial(&probs);
        u32::try_from(token).map_err(|_| SamplingError::VocabOverflow)
    }

    pub fn sample_greedy(&self, logits: &Tensor) -> Result<u32> {
        let logits = extract_logits(logits)?;
        u32::try_from(argmax(&logits)).map_err(|_| SamplingError::VocabOverflow)
    }

    pub fn sample_with_temp(&mut self, logits: &Tensor, temperature: f32) -> Result<u32> {
        if temperature == 0.0 {
            return self.sample_greedy(logits);
        }
        let mut logits = extract_logits(logits)?;
        apply_temperature(&mut logits, temperature);
        let probs = softmax(&logits);
        let token = self.sample_multinomial(&probs);
        u32::try_from(token).map_err(|_| SamplingError::VocabOverflow)
    }

    pub fn penalize(&self, logits: &mut [f32], tokens: &[u32], penalty: f32) {
        apply_repetition_penalty(logits, tokens, penalty);
    }

    fn sample_multinomial(&mut self, probs: &[f32]) -> usize {
        let r: f32 = self.rng.f32();
        let mut cumsum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }
        probs.len() - 1
    }
}

fn extract_logits(tensor: &Tensor) -> Result<Vec<f32>> {
    let shape = tensor.shape().to_vec();
    let vocab_size = match shape.len() {
        1 => shape[0],
        2 if shape[0] == 1 => shape[1],
        _ => return Err(SamplingError::InvalidShape(shape)),
    };

    let t = tensor.contiguous()?;
    let mut logits = Vec::with_capacity(vocab_size);

    if tensor.dims() == 1 {
        for i in 0..vocab_size {
            logits.push(t.get::<f32>(&[i])?);
        }
    } else {
        for i in 0..vocab_size {
            logits.push(t.get::<f32>(&[0, i])?);
        }
    }

    Ok(logits)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum.is_finite() && sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    probs
}

fn apply_temperature(logits: &mut [f32], temperature: f32) {
    let inv_temp = 1.0 / temperature;
    for logit in logits.iter_mut() {
        *logit *= inv_temp;
    }
}

fn apply_top_k(logits: &mut [f32], k: u32) {
    let k = k as usize;
    if k == 0 || k >= logits.len() {
        return;
    }
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.select_nth_unstable_by(k - 1, |&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = logits[indices[k - 1]];
    for logit in logits.iter_mut() {
        if *logit < threshold {
            *logit = f32::NEG_INFINITY;
        }
    }
}

fn apply_top_p(logits: &mut [f32], p: f32) {
    if p >= 1.0 {
        return;
    }

    let n = logits.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let max_logit = logits[indices[0]];
    let mut exp_vals = Vec::with_capacity(n);
    let mut exp_sum = 0.0;
    for &idx in &indices {
        let v = (logits[idx] - max_logit).exp();
        exp_vals.push(v);
        exp_sum += v;
    }

    let mut cumsum = 0.0;
    let mut cutoff = n;
    if exp_sum.is_finite() && exp_sum > 0.0 {
        for (i, &ev) in exp_vals.iter().enumerate() {
            let prob = ev / exp_sum;
            cumsum += prob;
            if cumsum > p {
                cutoff = i + 1;
                break;
            }
        }
    }

    for &idx in &indices[cutoff..] {
        logits[idx] = f32::NEG_INFINITY;
    }
}

fn apply_repetition_penalty(logits: &mut [f32], tokens: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() <= f32::EPSILON {
        return;
    }
    for &token in tokens {
        let idx = token as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(idx, _)| idx)
}

#[cfg(test)]
mod tests {
    use xllm_tensor::DType;

    use super::*;

    #[test]
    fn test_greedy_returns_argmax() {
        let data = vec![-2.0, 3.0, 5.0, -1.0, 0.0];
        let logits = Tensor::from_slice(&data, &[5]).unwrap();
        let sampler = Sampler::new(42);
        let token = sampler.sample_greedy(&logits).unwrap();
        assert_eq!(token, 2);
    }

    #[test]
    fn test_temperature_zero_falls_back_to_greedy() {
        let data: Vec<f32> = (0..100u8).map(f32::from).collect();
        let logits = Tensor::from_slice(&data, &[100]).unwrap();
        let mut sampler = Sampler::new(42);
        let params = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 42,
        };
        let token = sampler.sample(&logits, &params).unwrap();
        assert_eq!(token, 99);
    }

    #[test]
    fn test_top_k_filtering() {
        let data = vec![-5.0, -3.0, 10.0, 8.0, 4.0, -1.0, 2.0];
        let logits = Tensor::from_slice(&data, &[7]).unwrap();
        let mut logits = extract_logits(&logits).unwrap();
        apply_top_k(&mut logits, 3);
        let non_inf = logits.iter().filter(|&&v| v.is_finite()).count();
        assert_eq!(non_inf, 3);
    }

    #[test]
    fn test_top_k_with_k_greater_than_len() {
        let data = vec![-5.0, -3.0, 10.0, 8.0];
        let mut logits = data.clone();
        apply_top_k(&mut logits, 100);
        assert_eq!(logits, data);
    }

    #[test]
    fn test_top_p_nucleus_filtering() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let mut logits = data;
        apply_top_p(&mut logits, 0.5);
        let non_inf = logits.iter().filter(|&&v| v.is_finite()).count();
        assert!(non_inf < 5);
        assert!(non_inf >= 1);
    }

    #[test]
    fn test_top_p_disabled_at_one() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let mut logits = data.clone();
        apply_top_p(&mut logits, 1.0);
        assert_eq!(logits, data);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![-2.0, 3.0, 5.0, -1.0, 0.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_all_inf() {
        let logits = vec![f32::NEG_INFINITY; 5];
        let probs = softmax(&logits);
        assert!(probs.iter().all(|&p| p.is_nan()));
    }

    #[test]
    fn test_softmax_empty() {
        let probs = softmax(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_repetition_penalty_penalizes_specified_tokens() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        apply_repetition_penalty(&mut logits, &[2], 2.0);
        assert!((logits[2] - 1.5).abs() < 1e-5);
        assert!((logits[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_repetition_penalty_negative_logits() {
        let mut logits = vec![-1.0, 2.0, -3.0, 4.0];
        apply_repetition_penalty(&mut logits, &[0, 2], 2.0);
        assert!((logits[0] - (-2.0)).abs() < 1e-5);
        assert!((logits[2] - (-6.0)).abs() < 1e-5);
    }

    #[test]
    fn test_invalid_shape_returns_error() {
        let logits = Tensor::zeros(&[2, 3, 4], DType::F32);
        let mut sampler = Sampler::new(42);
        let params = SamplingParams::default();
        let result = sampler.sample(&logits, &params);
        assert!(matches!(result, Err(SamplingError::InvalidShape(_))));
    }

    #[test]
    fn test_all_neg_inf_logits_error() {
        let data = vec![f32::NEG_INFINITY; 10];
        let logits = Tensor::from_slice(&data, &[10]).unwrap();
        let mut sampler = Sampler::new(42);
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 42,
        };
        let result = sampler.sample(&logits, &params);
        assert!(matches!(result, Err(SamplingError::NoValidTokens)));
    }

    #[test]
    fn test_different_seeds_produce_different_sequences() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let logits = Tensor::from_slice(&data, &[10]).unwrap();

        let mut sampler_a = Sampler::new(42);
        let mut sampler_b = Sampler::new(999);

        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 0,
        };

        let mut results_a = Vec::new();
        let mut results_b = Vec::new();
        for _ in 0..20 {
            results_a.push(sampler_a.sample(&logits, &params).unwrap());
            results_b.push(sampler_b.sample(&logits, &params).unwrap());
        }

        assert_ne!(results_a, results_b);
    }

    #[test]
    fn test_same_seed_reproduces_results() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let logits = Tensor::from_slice(&data, &[10]).unwrap();

        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 123,
        };

        let mut sampler_1 = Sampler::new(123);
        let mut sampler_2 = Sampler::new(123);

        let mut results_1 = Vec::new();
        let mut results_2 = Vec::new();
        for _ in 0..20 {
            results_1.push(sampler_1.sample(&logits, &params).unwrap());
            results_2.push(sampler_2.sample(&logits, &params).unwrap());
        }

        assert_eq!(results_1, results_2);
    }

    #[test]
    fn test_sample_with_temp_greedy_fallback() {
        let data = vec![-5.0, 100.0, -5.0];
        let logits = Tensor::from_slice(&data, &[3]).unwrap();
        let mut sampler = Sampler::new(42);
        let token = sampler.sample_with_temp(&logits, 0.0).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn test_argmax_returns_correct_index() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[0.0]), 0);
        assert_eq!(argmax(&[f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY]), 1);
    }

    #[test]
    fn test_2d_logits_tensor() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let logits = Tensor::from_slice(&data, &[1, 5]).unwrap();
        let sampler = Sampler::new(42);
        let token = sampler.sample_greedy(&logits).unwrap();
        assert_eq!(token, 4);
    }

    #[test]
    fn test_penalize_modifies_logits() {
        let mut logits = vec![1.0, 1.0, 1.0, 1.0];
        let sampler = Sampler::new(42);
        sampler.penalize(&mut logits, &[0, 2], 2.0);
        assert!((logits[0] - 0.5).abs() < 1e-5);
        assert!((logits[1] - 1.0).abs() < 1e-5);
        assert!((logits[2] - 0.5).abs() < 1e-5);
        assert!((logits[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_top_k_equals_one_is_greedy() {
        let data = vec![0.0, 100.0, 0.0];
        let logits = Tensor::from_slice(&data, &[3]).unwrap();
        let mut sampler = Sampler::new(42);
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 0,
        };
        let token = sampler.sample(&logits, &params).unwrap();
        assert_eq!(token, 1);
    }
}
