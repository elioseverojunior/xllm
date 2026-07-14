// SPDX-FileCopyrightText: 2026 XLLM contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use std::sync::Arc;

use xllm_model::{GGUFValue, Model, ModelError};
use xllm_tensor::{DType, Tensor, TensorError};

// ---------------------------------------------------------------------------
// ForwardResult
// ---------------------------------------------------------------------------

/// Output of a single forward pass — logits for the last token.
#[derive(Debug)]
pub struct ForwardResult {
    pub logits: Tensor,
    pub n_past: usize,
}

// ---------------------------------------------------------------------------
// ContextError
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    #[error("tensor error: {0}")]
    TensorError(#[from] TensorError),
    #[error("model error: {0}")]
    ModelError(#[from] ModelError),
    #[error("tensor '{0}' not found in model")]
    WeightNotFound(String),
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),
    #[error("context length exceeded (max: {max}, requested: {requested})")]
    ContextLengthExceeded { max: usize, requested: usize },
}

pub type Result<T> = std::result::Result<T, ContextError>;

// ---------------------------------------------------------------------------
// ModelConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

impl ModelConfig {
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ---------------------------------------------------------------------------
// Metadata helpers
// ---------------------------------------------------------------------------

fn get_meta_u64(model: &Model, key: &str, default: u64) -> u64 {
    model
        .metadata_value(key)
        .and_then(|v| match v {
            GGUFValue::Uint64(u) => Some(*u),
            GGUFValue::Int32(i) => Some(*i as u64),
            GGUFValue::Uint32(u) => Some(u64::from(*u)),
            GGUFValue::Int64(i) => Some(*i as u64),
            _ => None,
        })
        .unwrap_or(default)
}

fn get_meta_f32(model: &Model, key: &str, default: f32) -> f32 {
    model
        .metadata_value(key)
        .and_then(|v| match v {
            GGUFValue::Float32(f) => Some(*f),
            GGUFValue::Float64(f) => Some(*f as f32),
            _ => None,
        })
        .unwrap_or(default)
}

impl ModelConfig {
    /// Extract model configuration from GGUF metadata.
    /// Works with architectures like "llama", "mistral", etc.
    ///
    /// # Errors
    ///
    /// Returns `UnsupportedArchitecture` if the model lacks an architecture
    /// metadata entry.
    pub fn from_model(model: &Model) -> Result<Self> {
        let arch = model
            .architecture()
            .ok_or_else(|| ContextError::UnsupportedArchitecture("unknown".to_string()))?
            .to_string();
        let prefix = format!("{arch}.");

        let n_heads = get_meta_u64(model, &format!("{prefix}attention.head_count"), 32) as usize;
        let n_kv_heads = get_meta_u64(
            model,
            &format!("{prefix}attention.head_count_kv"),
            n_heads as u64,
        ) as usize;

        Ok(Self {
            architecture: arch,
            vocab_size: get_meta_u64(model, &format!("{prefix}vocab_size"), 32000) as usize,
            hidden_size: get_meta_u64(model, &format!("{prefix}embedding_length"), 4096) as usize,
            intermediate_size: get_meta_u64(model, &format!("{prefix}feed_forward_length"), 11008)
                as usize,
            num_hidden_layers: get_meta_u64(model, &format!("{prefix}block_count"), 32) as usize,
            num_attention_heads: n_heads,
            num_kv_heads: n_kv_heads,
            max_position_embeddings: get_meta_u64(model, &format!("{prefix}context_length"), 2048)
                as usize,
            rms_norm_eps: get_meta_f32(
                model,
                &format!("{prefix}attention.layer_norm_rms_epsilon"),
                1e-5,
            ),
            rope_theta: get_meta_f32(model, &format!("{prefix}rope.freq_base"), 10000.0),
            bos_token_id: model
                .metadata_value("tokenizer.ggml.bos_id")
                .and_then(|v| match v {
                    GGUFValue::Int32(i) => Some(*i as u32),
                    _ => None,
                })
                .unwrap_or(1),
            eos_token_id: model
                .metadata_value("tokenizer.ggml.eos_id")
                .and_then(|v| match v {
                    GGUFValue::Int32(i) => Some(*i as u32),
                    _ => None,
                })
                .unwrap_or(2),
        })
    }
}

// ---------------------------------------------------------------------------
// InferenceContext
// ---------------------------------------------------------------------------

pub struct InferenceContext {
    config: ModelConfig,
    model: Arc<Model>,
    k_cache: Vec<Tensor>,
    v_cache: Vec<Tensor>,
    n_past: usize,
}

impl InferenceContext {
    /// Create a new inference context from a loaded GGUF model.
    ///
    /// # Errors
    ///
    /// Returns `ContextError` if the model config cannot be parsed from
    /// metadata, or if KV cache allocation fails.
    pub fn new(model: Model) -> Result<Self> {
        let config = ModelConfig::from_model(&model)?;
        let model = Arc::new(model);

        let max_seq_len = config.max_position_embeddings;
        let n_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim();

        let mut k_cache = Vec::with_capacity(config.num_hidden_layers);
        let mut v_cache = Vec::with_capacity(config.num_hidden_layers);

        for _ in 0..config.num_hidden_layers {
            let cache_shape = [max_seq_len, n_kv_heads * head_dim];
            k_cache.push(Tensor::zeros(&cache_shape, DType::F32));
            v_cache.push(Tensor::zeros(&cache_shape, DType::F32));
        }

        Ok(Self {
            config,
            model,
            k_cache,
            v_cache,
            n_past: 0,
        })
    }

    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        &self.config
    }

    #[must_use]
    pub const fn n_past(&self) -> usize {
        self.n_past
    }

    // -----------------------------------------------------------------------
    // Weight loading
    // -----------------------------------------------------------------------

    fn get_weight(&self, name: &str) -> Result<Tensor> {
        match self.model.tensor(name) {
            Ok(t) => Ok(t),
            Err(ModelError::TensorNotFound(_)) => {
                Err(ContextError::WeightNotFound(name.to_string()))
            }
            Err(e) => Err(ContextError::ModelError(e)),
        }
    }

    // -----------------------------------------------------------------------
    // Embedding
    // -----------------------------------------------------------------------

    fn embed(tokens: &[u32], weight: &Tensor) -> Result<Tensor> {
        let seq_len = tokens.len();
        let hidden_size = weight.shape()[1];
        let mut result = Tensor::zeros(&[seq_len, hidden_size], DType::F32);
        for (pos, &token_id) in tokens.iter().enumerate() {
            for j in 0..hidden_size {
                let val: f32 = weight.get(&[token_id as usize, j])?;
                result.set(&[pos, j], val)?;
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Forward pass
    // -----------------------------------------------------------------------

    /// Run forward pass for a sequence of token IDs.
    /// Returns logits for the last token position.
    ///
    /// # Errors
    ///
    /// Returns `ContextLengthExceeded` if the total context length (past + new)
    /// exceeds the model's `max_position_embeddings`. Returns `WeightNotFound`
    /// if a required weight tensor is missing. Returns `TensorError` for
    /// dimension mismatches and other tensor-level errors.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<ForwardResult> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Ok(ForwardResult {
                logits: Tensor::zeros(&[self.config.vocab_size], DType::F32),
                n_past: self.n_past,
            });
        }

        let total_len = self.n_past + seq_len;
        if total_len > self.config.max_position_embeddings {
            return Err(ContextError::ContextLengthExceeded {
                max: self.config.max_position_embeddings,
                requested: total_len,
            });
        }

        let embed_weight = self.get_weight("token_embd.weight")?;
        let mut x = Self::embed(tokens, &embed_weight)?;

        for layer_idx in 0..self.config.num_hidden_layers {
            x = self.process_layer(layer_idx, &x, seq_len)?;
        }

        let output_norm_w = self.get_weight("output_norm.weight")?;
        let x_normed = rms_norm_1d(&x, &output_norm_w, self.config.rms_norm_eps)?;

        let output_w = self.get_weight("output.weight")?;
        let output_w_t = output_w.transpose(0, 1)?;
        let logits_full = x_normed.matmul(&output_w_t)?;

        let last_logits = Self::last_token_logits(&logits_full, seq_len, self.config.vocab_size)?;

        self.n_past += seq_len;

        Ok(ForwardResult {
            logits: last_logits,
            n_past: self.n_past,
        })
    }

    fn process_layer(&mut self, layer_idx: usize, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        let blk = format!("blk.{layer_idx}");
        let hidden_size = self.config.hidden_size;

        let attn_out = self.attention_block(layer_idx, &blk, x, seq_len)?;
        let mut result = x.clone();
        Self::add_residual(&mut result, &attn_out, seq_len, hidden_size)?;

        let ffn_out = self.ffn_block(&blk, &result, seq_len)?;
        Self::add_residual(&mut result, &ffn_out, seq_len, hidden_size)?;

        Ok(result)
    }

    fn attention_block(
        &mut self,
        layer_idx: usize,
        blk: &str,
        x: &Tensor,
        seq_len: usize,
    ) -> Result<Tensor> {
        let eps = self.config.rms_norm_eps;
        let n_heads = self.config.num_attention_heads;
        let n_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let theta = self.config.rope_theta;

        let attn_norm_w = self.get_weight(&format!("{blk}.attn_norm.weight"))?;
        let h = rms_norm_1d(x, &attn_norm_w, eps)?;

        let q_w = self.get_weight(&format!("{blk}.attn_q.weight"))?;
        let k_w = self.get_weight(&format!("{blk}.attn_k.weight"))?;
        let v_w = self.get_weight(&format!("{blk}.attn_v.weight"))?;
        let o_w = self.get_weight(&format!("{blk}.attn_output.weight"))?;

        let q_w_t = q_w.transpose(0, 1)?;
        let k_w_t = k_w.transpose(0, 1)?;
        let v_w_t = v_w.transpose(0, 1)?;
        let o_w_t = o_w.transpose(0, 1)?;

        let q = h.matmul(&q_w_t)?;
        let k = h.matmul(&k_w_t)?;
        let v = h.matmul(&v_w_t)?;

        let q_reshaped = q.reshape(&[seq_len, n_heads, head_dim])?;
        let k_reshaped = k.reshape(&[seq_len, n_kv_heads, head_dim])?;

        let q_rope = apply_rope(&q_reshaped, self.n_past, theta, head_dim)?;
        let k_rope = apply_rope(&k_reshaped, self.n_past, theta, head_dim)?;

        let q_flat = q_rope.reshape(&[seq_len, n_heads * head_dim])?;
        let k_flat = k_rope.reshape(&[seq_len, n_kv_heads * head_dim])?;

        let kv_width = n_kv_heads * head_dim;
        Self::update_kv_cache(
            &mut self.k_cache[layer_idx],
            &mut self.v_cache[layer_idx],
            &k_flat,
            &v,
            seq_len,
            self.n_past,
            kv_width,
        )?;

        let cache_len = self.n_past + seq_len;
        let k_cached = self.k_cache[layer_idx].slice(&[0..cache_len, 0..kv_width])?;
        let v_cached = self.v_cache[layer_idx].slice(&[0..cache_len, 0..kv_width])?;

        let attn_out = attention(&q_flat, &k_cached, &v_cached, n_heads, n_kv_heads, head_dim)?;
        Ok(attn_out.matmul(&o_w_t)?)
    }

    fn ffn_block(&self, blk: &str, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        let eps = self.config.rms_norm_eps;
        let intermediate_size = self.config.intermediate_size;

        let ffn_norm_w = self.get_weight(&format!("{blk}.ffn_norm.weight"))?;
        let h2 = rms_norm_1d(x, &ffn_norm_w, eps)?;

        let gate_w = self.get_weight(&format!("{blk}.ffn_gate.weight"))?;
        let up_w = self.get_weight(&format!("{blk}.ffn_up.weight"))?;
        let down_w = self.get_weight(&format!("{blk}.ffn_down.weight"))?;

        let gate_w_t = gate_w.transpose(0, 1)?;
        let up_w_t = up_w.transpose(0, 1)?;
        let down_w_t = down_w.transpose(0, 1)?;

        let gate = h2.matmul(&gate_w_t)?;
        let up = h2.matmul(&up_w_t)?;

        let silu_gate = silu(&gate)?;

        let mut ffn_input = Tensor::zeros(&[seq_len, intermediate_size], DType::F32);
        for pos in 0..seq_len {
            for j in 0..intermediate_size {
                let g: f32 = silu_gate.get(&[pos, j])?;
                let u: f32 = up.get(&[pos, j])?;
                ffn_input.set(&[pos, j], g * u)?;
            }
        }

        Ok(ffn_input.matmul(&down_w_t)?)
    }

    fn update_kv_cache(
        k_cache: &mut Tensor,
        v_cache: &mut Tensor,
        k_flat: &Tensor,
        v: &Tensor,
        seq_len: usize,
        n_past: usize,
        kv_width: usize,
    ) -> Result<()> {
        for pos in 0..seq_len {
            let cp = n_past + pos;
            for h in 0..kv_width {
                let kv: f32 = k_flat.get(&[pos, h])?;
                k_cache.set(&[cp, h], kv)?;
                let vv: f32 = v.get(&[pos, h])?;
                v_cache.set(&[cp, h], vv)?;
            }
        }
        Ok(())
    }

    fn add_residual(
        tensor: &mut Tensor,
        delta: &Tensor,
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<()> {
        for pos in 0..seq_len {
            for j in 0..hidden_size {
                let d: f32 = delta.get(&[pos, j])?;
                let r: f32 = tensor.get(&[pos, j])?;
                tensor.set(&[pos, j], d + r)?;
            }
        }
        Ok(())
    }

    fn last_token_logits(
        logits_full: &Tensor,
        seq_len: usize,
        vocab_size: usize,
    ) -> Result<Tensor> {
        let last_pos = seq_len - 1;
        let mut last_logits = Tensor::zeros(&[vocab_size], DType::F32);
        for j in 0..vocab_size {
            let val: f32 = logits_full.get(&[last_pos, j])?;
            last_logits.set(&[j], val)?;
        }
        Ok(last_logits)
    }
}

// ---------------------------------------------------------------------------
// Operator helpers
// ---------------------------------------------------------------------------

/// RMS Normalisation: `x / sqrt(mean(x²) + eps) * weight`
///
/// `x`: `[seq_len, hidden_size]`
/// `weight`: `[hidden_size]` — per-element scale factors
fn rms_norm_1d(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let shape = x.shape();
    let seq_len = shape[0];
    let hidden_size = shape[1];
    let mut result = Tensor::zeros(&[seq_len, hidden_size], DType::F32);

    for i in 0..seq_len {
        let mut ss = 0.0f32;
        for j in 0..hidden_size {
            let val: f32 = x.get(&[i, j])?;
            ss += val * val;
        }
        let rms = (ss / hidden_size as f32 + eps).sqrt();
        let rms_inv = 1.0 / rms;

        for j in 0..hidden_size {
            let val: f32 = x.get(&[i, j])?;
            let w: f32 = weight.get(&[j])?;
            result.set(&[i, j], val * rms_inv * w)?;
        }
    }
    Ok(result)
}

/// Rotary Position Embedding (`RoPE`)
///
/// `x`: `[seq_len, n_heads, head_dim]` — Q or K tensor
/// `start_pos`: absolute position offset for the first token
fn apply_rope(x: &Tensor, start_pos: usize, theta: f32, head_dim: usize) -> Result<Tensor> {
    let shape = x.shape();
    let seq_len = shape[0];
    let n_heads = shape[1];

    let mut result = Tensor::zeros(&[seq_len, n_heads, head_dim], DType::F32);

    for pos in 0..seq_len {
        let abs_pos = (start_pos + pos) as f32;
        for h in 0..n_heads {
            for d in (0..head_dim).step_by(2) {
                let freq = 1.0 / theta.powf(d as f32 / head_dim as f32);
                let (sin, cos) = (abs_pos * freq).sin_cos();

                let x1: f32 = x.get(&[pos, h, d])?;
                let x2: f32 = x.get(&[pos, h, d + 1])?;

                let y1 = x1 * cos - x2 * sin;
                let y2 = x2 * cos + x1 * sin;

                result.set(&[pos, h, d], y1)?;
                result.set(&[pos, h, d + 1], y2)?;
            }
        }
    }

    Ok(result)
}

/// Scaled dot-product attention with Grouped-Query Attention support.
///
/// `q`: `[seq_len, n_heads * head_dim]`
/// `k`: `[cache_len, n_kv_heads * head_dim]`
/// `v`: `[cache_len, n_kv_heads * head_dim]`
/// Returns `[seq_len, n_heads * head_dim]`
fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let seq_len = q.shape()[0];
    let cache_len = k.shape()[0];
    let n_gqa = n_heads / n_kv_heads;

    let mut output = Tensor::zeros(&[seq_len, n_heads * head_dim], DType::F32);
    let inv_scale = 1.0 / (head_dim as f32).sqrt();

    for i in 0..seq_len {
        for h in 0..n_heads {
            let kv_h = h / n_gqa;

            let mut scores = vec![0.0f32; cache_len];
            for j in 0..cache_len {
                let mut s = 0.0f32;
                for d in 0..head_dim {
                    let qv: f32 = q.get(&[i, h * head_dim + d])?;
                    let kv_val: f32 = k.get(&[j, kv_h * head_dim + d])?;
                    s += qv * kv_val;
                }
                scores[j] = s * inv_scale;
            }

            // Softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                exp_sum += *s;
            }
            let inv_sum = 1.0 / exp_sum;
            for s in &mut scores {
                *s *= inv_sum;
            }

            // Weighted sum of V
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for j in 0..cache_len {
                    let vv: f32 = v.get(&[j, kv_h * head_dim + d])?;
                    val += scores[j] * vv;
                }
                output.set(&[i, h * head_dim + d], val)?;
            }
        }
    }

    Ok(output)
}

/// `SiLU` activation: `x * sigmoid(x)`
fn silu(x: &Tensor) -> Result<Tensor> {
    let shape = x.shape();
    let total = x.size();
    let mut result = Tensor::zeros(shape, DType::F32);
    for flat in 0..total {
        let mut idx = flat;
        let mut indices = vec![0usize; shape.len()];
        for d in (0..shape.len()).rev() {
            indices[d] = idx % shape[d];
            idx /= shape[d];
        }
        let val: f32 = x.get(&indices)?;
        let sigmoid = 1.0 / (1.0 + (-val).exp());
        result.set(&indices, val * sigmoid)?;
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test helpers: GGUF binary construction
    // -----------------------------------------------------------------------

    const GGUF_ALIGNMENT: u64 = 32;

    fn align_up(val: u64, alignment: u64) -> u64 {
        (val + alignment - 1) & !(alignment - 1)
    }

    /// Write GGUF bytes to a temp file and return the path.
    fn write_temp_gguf(data: &[u8]) -> std::path::PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "test_context_{}_{}.gguf",
            std::process::id(),
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::write(&path, data).unwrap();
        path
    }

    /// Create a GGUF binary with a minimal llama config and no tensors.
    fn create_config_gguf(
        arch: &str,
        kv_pairs: &[(&str, (u32, Vec<u8>))], // (key, (value_type, value_bytes))
    ) -> Vec<u8> {
        let mut buf = Vec::new();

        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&((kv_pairs.len() + 1) as u64).to_le_bytes());

        // Architecture KV pair
        let arch_key = b"general.architecture";
        buf.extend_from_slice(&(arch_key.len() as u64).to_le_bytes());
        buf.extend_from_slice(arch_key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // String type
        buf.extend_from_slice(&(arch.len() as u64).to_le_bytes());
        buf.extend_from_slice(arch.as_bytes());

        // Additional KV pairs
        for (key, (ty, val_bytes)) in kv_pairs {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&ty.to_le_bytes());
            buf.extend_from_slice(val_bytes);
        }

        buf
    }

    /// Create a GGUF binary with a single f32 tensor.
    #[allow(dead_code)]
    fn create_tensor_gguf(
        tensor_name: &str,
        shape: &[usize], // standard order (outermost first)
        data: &[f32],
        kv_pairs: &[(&str, (u32, Vec<u8>))],
    ) -> Vec<u8> {
        let mut buf = Vec::new();

        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        buf.extend_from_slice(&(kv_pairs.len() as u64).to_le_bytes());

        // KV pairs first
        for (key, (ty, val_bytes)) in kv_pairs {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&ty.to_le_bytes());
            buf.extend_from_slice(val_bytes);
        }

        // Tensor info: GGUF stores dims reversed (innermost first)
        buf.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        buf.extend_from_slice(tensor_name.as_bytes());
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &d in shape.iter().rev() {
            buf.extend_from_slice(&(d as u64).to_le_bytes());
        }
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32

        let data_offset_hint = (buf.len() + 8) as u64;
        let aligned_offset = align_up(data_offset_hint, GGUF_ALIGNMENT);
        buf.extend_from_slice(&aligned_offset.to_le_bytes());

        // Pad to aligned offset
        while buf.len() < aligned_offset as usize {
            buf.push(0);
        }

        for &v in data {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        buf
    }

    fn u64_bytes(v: u64) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    fn f32_bytes(v: f32) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    fn i32_bytes(v: i32) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    #[allow(dead_code)]
    fn str_bytes(s: &str) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
        b
    }

    // -----------------------------------------------------------------------
    // Tests: ModelConfig::from_model
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_config_from_model() {
        let arch = "llama";
        let kv_pairs: Vec<(&str, (u32, Vec<u8>))> = vec![
            ("llama.vocab_size", (10u32, u64_bytes(32000))),
            ("llama.embedding_length", (10u32, u64_bytes(4096))),
            ("llama.feed_forward_length", (10u32, u64_bytes(11008))),
            ("llama.block_count", (10u32, u64_bytes(32))),
            ("llama.attention.head_count", (10u32, u64_bytes(32))),
            ("llama.attention.head_count_kv", (10u32, u64_bytes(8))),
            ("llama.context_length", (10u32, u64_bytes(2048))),
            (
                "llama.attention.layer_norm_rms_epsilon",
                (6u32, f32_bytes(1e-5)),
            ),
            ("llama.rope.freq_base", (6u32, f32_bytes(10000.0))),
            ("tokenizer.ggml.bos_id", (5u32, i32_bytes(1))),
            ("tokenizer.ggml.eos_id", (5u32, i32_bytes(2))),
        ];

        let gguf = create_config_gguf(arch, &kv_pairs);
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let config = ModelConfig::from_model(&model).unwrap();

        assert_eq!(config.architecture, "llama");
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.intermediate_size, 11008);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.max_position_embeddings, 2048);
        assert!((config.rms_norm_eps - 1e-5).abs() < 1e-10);
        assert!((config.rope_theta - 10000.0).abs() < 1e-6);
        assert_eq!(config.bos_token_id, 1);
        assert_eq!(config.eos_token_id, 2);
        assert_eq!(config.head_dim(), 128); // 4096 / 32

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_model_config_defaults() {
        // Minimal model with only architecture and no depth-specific metadata
        let arch = "llama";
        let kv_pairs: Vec<(&str, (u32, Vec<u8>))> = vec![];

        let gguf = create_config_gguf(arch, &kv_pairs);
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let config = ModelConfig::from_model(&model).unwrap();

        assert_eq!(config.architecture, "llama");
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads, 32); // defaults to n_heads
        assert_eq!(config.head_dim(), 128);

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_model_config_missing_architecture() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        let path = write_temp_gguf(&buf);
        let model = Model::load(&path).unwrap();
        let err = ModelConfig::from_model(&model).unwrap_err();
        assert!(matches!(err, ContextError::UnsupportedArchitecture(_)));

        std::fs::remove_file(path).unwrap();
    }

    // -----------------------------------------------------------------------
    // Tests: RMS Norm
    // -----------------------------------------------------------------------

    #[test]
    fn test_rms_norm_1d_basic() {
        // x = [1, 2, 3], weight = [1, 1, 1], eps = 1e-6
        // mean(x^2) = (1+4+9)/3 = 14/3 ≈ 4.6667
        // rms = sqrt(4.6667 + 1e-6) ≈ 2.16025
        // output = x / rms ≈ [0.4629, 0.9258, 1.3887]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3]).unwrap();
        let w = Tensor::from_slice(&[1.0f32, 1.0, 1.0], &[3]).unwrap();
        let result = rms_norm_1d(&x, &w, 1e-6).unwrap();

        let expected: Vec<f32> = [0.462_910_06f32, 0.925_820_1, 1.388_730_2].to_vec();
        for j in 0..3 {
            let val: f32 = result.get(&[0, j]).unwrap();
            assert!(
                (val - expected[j]).abs() < 1e-5,
                "mismatch at {j}: {val} vs {}",
                expected[j]
            );
        }
    }

    #[test]
    fn test_rms_norm_1d_with_weight() {
        // x = [1, 2, -1], weight = [0.5, 2.0, 1.5], eps = 1e-6
        // mean(x^2) = (1+4+1)/3 = 2.0
        // rms = sqrt(2.0) ≈ 1.4142
        // normed = x / rms ≈ [0.7071, 1.4142, -0.7071]
        // output = normed * weight ≈ [0.3536, 2.8284, -1.0607]
        let x = Tensor::from_slice(&[1.0f32, 2.0, -1.0], &[1, 3]).unwrap();
        let w = Tensor::from_slice(&[0.5f32, 2.0, 1.5], &[3]).unwrap();
        let result = rms_norm_1d(&x, &w, 1e-6).unwrap();

        let rms = (2.0f32).sqrt();
        let inv = 1.0 / rms;
        let expected: Vec<f32> = [1.0 * inv * 0.5, 2.0 * inv * 2.0, -(inv * 1.5)].to_vec();

        for j in 0..3 {
            let val: f32 = result.get(&[0, j]).unwrap();
            assert!(
                (val - expected[j]).abs() < 1e-5,
                "mismatch at {j}: {val} vs {}",
                expected[j]
            );
        }
    }

    #[test]
    fn test_rms_norm_1d_multi_row() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let w = Tensor::from_slice(&[1.0f32, 1.0, 1.0], &[3]).unwrap();
        let result = rms_norm_1d(&x, &w, 1e-6).unwrap();

        assert_eq!(result.shape(), &[2, 3]);

        // Row 0: [1,2,3]
        let rms0 = ((14.0f32 / 3.0) + 1e-6).sqrt();
        let inv0 = 1.0 / rms0;
        assert!((result.get::<f32>(&[0, 0]).unwrap() - 1.0 * inv0).abs() < 1e-5);

        // Row 1: [4,5,6]
        let rms1 = ((16.0f64 + 25.0 + 36.0) / 3.0 + 1e-6).sqrt() as f32;
        let inv1 = 1.0 / rms1;
        assert!((result.get::<f32>(&[1, 0]).unwrap() - 4.0 * inv1).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Tests: RoPE
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_rope_basic() {
        // Single position, single head, head_dim=4, start_pos=0
        // x = [1, 0, 0, 1]
        // theta = 10000.0
        // freq[0] = 1 / 10000^(0/4) = 1
        // freq[2] = 1 / 10000^(2/4) = 1 / 10000^0.5 = 1/100 = 0.01
        // pos=0: cos(0)=1, sin(0)=0 => no change
        // y = [1*1-0*0, 0*1+1*0, 0*1-1*0, 1*1+0*0] = [1, 0, 0, 1]
        let x = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[1, 1, 4]).unwrap();
        let result = apply_rope(&x, 0, 10000.0, 4).unwrap();

        assert_eq!(result.shape(), &[1, 1, 4]);
        assert!((result.get::<f32>(&[0, 0, 0]).unwrap() - 1.0).abs() < 1e-6);
        assert!((result.get::<f32>(&[0, 0, 1]).unwrap() - 0.0).abs() < 1e-6);
        assert!((result.get::<f32>(&[0, 0, 2]).unwrap() - 0.0).abs() < 1e-6);
        assert!((result.get::<f32>(&[0, 0, 3]).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_rope_nonzero_pos() {
        // head_dim=2, theta=10000
        // freq[0] = 1
        // pos=1: cos(1)=0.5403, sin(1)=0.8415
        // x = [1, 0]
        // y1 = 1*0.5403 - 0*0.8415 = 0.5403
        // y2 = 0*0.5403 + 1*0.8415 = 0.8415
        let x = Tensor::from_slice(&[1.0f32, 0.0], &[1, 1, 2]).unwrap();
        let result = apply_rope(&x, 1, 10000.0, 2).unwrap();

        let cos = 1.0f32.cos();
        let sin = 1.0f32.sin();
        assert!(
            (result.get::<f32>(&[0, 0, 0]).unwrap() - cos).abs() < 1e-5,
            "expected {cos}, got {}",
            result.get::<f32>(&[0, 0, 0]).unwrap()
        );
        assert!(
            (result.get::<f32>(&[0, 0, 1]).unwrap() - sin).abs() < 1e-5,
            "expected {sin}, got {}",
            result.get::<f32>(&[0, 0, 1]).unwrap()
        );
    }

    #[test]
    fn test_apply_rope_two_heads() {
        // head_dim=2, 2 heads, 1 position
        let data: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let x = Tensor::from_slice(&data, &[1, 2, 2]).unwrap();
        let result = apply_rope(&x, 0, 10000.0, 2).unwrap();

        // pos=0: cos=1, sin=0 => no change
        for i in 0..4 {
            let val: f32 = result.get(&[0, i / 2, i % 2]).unwrap();
            assert!((val - i as f32).abs() < 1e-5, "mismatch at {i}");
        }
    }

    // -----------------------------------------------------------------------
    // Tests: SiLU
    // -----------------------------------------------------------------------

    #[test]
    fn test_silu_zero() {
        let x = Tensor::from_slice(&[0.0f32], &[1, 1]).unwrap();
        let result = silu(&x).unwrap();
        let val: f32 = result.get(&[0, 0]).unwrap();
        // silu(0) = 0 * 0.5 = 0
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_silu_positive() {
        let x = Tensor::from_slice(&[2.0f32], &[1, 1]).unwrap();
        let result = silu(&x).unwrap();
        let val: f32 = result.get(&[0, 0]).unwrap();
        // sigmoid(2) = 1/(1+e^-2) ≈ 0.8808
        // silu(2) = 2 * 0.8808 ≈ 1.7616
        let expected = 2.0 / (1.0 + (-2.0f32).exp());
        assert!((val - expected).abs() < 1e-5);
    }

    #[test]
    fn test_silu_negative() {
        let x = Tensor::from_slice(&[-1.0f32], &[1, 1]).unwrap();
        let result = silu(&x).unwrap();
        let val: f32 = result.get(&[0, 0]).unwrap();
        // sigmoid(-1) = 1/(1+e^1) ≈ 0.2689
        // silu(-1) = -1 * 0.2689 ≈ -0.2689
        let expected = -1.0 / (1.0 + 1.0f32.exp());
        assert!((val - expected).abs() < 1e-5);
    }

    #[test]
    fn test_silu_2d() {
        let x = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[2, 2]).unwrap();
        let result = silu(&x).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        for i in 0..4 {
            let val: f32 = result.get(&[i / 2, i % 2]).unwrap();
            let inp = i as f32;
            let sig = 1.0 / (1.0 + (-inp).exp());
            let expected = inp * sig;
            assert!((val - expected).abs() < 1e-5, "mismatch at {i}");
        }
    }

    // -----------------------------------------------------------------------
    // Tests: Embed
    // -----------------------------------------------------------------------

    #[test]
    fn test_embed_1d() {
        // Embedding weight: vocab_size=4, hidden_size=3
        let weight_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let weight = Tensor::from_slice(&weight_data, &[4, 3]).unwrap();

        let _context = InferenceContext {
            config: ModelConfig {
                architecture: "test".to_string(),
                vocab_size: 4,
                hidden_size: 3,
                intermediate_size: 9,
                num_hidden_layers: 1,
                num_attention_heads: 1,
                num_kv_heads: 1,
                max_position_embeddings: 10,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                bos_token_id: 1,
                eos_token_id: 2,
            },
            model: Arc::new(
                // Load a minimal dummy model for the Arc
                // We use Model::load on a tiny GGUF with no tensors
                // Instead, we create a fake model... but we can't create Model
                // directly since its fields are private.
                //
                // We'll use Model::load on a minimal GGUF with architecture key.
                load_minimal_model(),
            ),
            k_cache: vec![],
            v_cache: vec![],
            n_past: 0,
        };

        let tokens = [0u32, 2u32];
        let result = InferenceContext::embed(&tokens, &weight).unwrap();
        assert_eq!(result.shape(), &[2, 3]);

        // Token 0: row 0 = [0, 1, 2]
        assert!((result.get::<f32>(&[0, 0]).unwrap() - 0.0).abs() < 1e-6);
        assert!((result.get::<f32>(&[0, 1]).unwrap() - 1.0).abs() < 1e-6);
        assert!((result.get::<f32>(&[0, 2]).unwrap() - 2.0).abs() < 1e-6);

        // Token 2: row 2 = [6, 7, 8]
        assert!((result.get::<f32>(&[1, 0]).unwrap() - 6.0).abs() < 1e-6);
        assert!((result.get::<f32>(&[1, 1]).unwrap() - 7.0).abs() < 1e-6);
        assert!((result.get::<f32>(&[1, 2]).unwrap() - 8.0).abs() < 1e-6);
    }

    fn load_minimal_model() -> Model {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 KV

        let arch_key = b"general.architecture";
        let arch_val = b"test";
        buf.extend_from_slice(&(arch_key.len() as u64).to_le_bytes());
        buf.extend_from_slice(arch_key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // String
        buf.extend_from_slice(&(arch_val.len() as u64).to_le_bytes());
        buf.extend_from_slice(arch_val);

        let path = write_temp_gguf(&buf);
        let model = Model::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        model
    }

    // -----------------------------------------------------------------------
    // Tests: Attention
    // -----------------------------------------------------------------------

    #[test]
    fn test_attention_single_query_head() {
        // Single query, single head, head_dim=2, cache_len=2
        // Q: [1, 0] at position 0
        // K: [[1, 0], [0, 1]]
        // V: [[1, 0], [0, 1]]
        // Scores: [1*1+0*0, 1*0+0*1] = [1, 0] / sqrt(2) ≈ [0.7071, 0]
        // Softmax: [e^0.7071/(e^0.7071+e^0), e^0/(...)]
        //   = [0.6698, 0.3302]
        // Output: 0.6698*[1,0] + 0.3302*[0,1] = [0.6698, 0.3302]

        let q = Tensor::from_slice(&[1.0f32, 0.0], &[1, 2]).unwrap();
        let k = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let v = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap();

        let result = attention(&q, &k, &v, 1, 1, 2).unwrap();
        assert_eq!(result.shape(), &[1, 2]);

        // Manual computation
        let scale = 1.0 / (2.0f32).sqrt();
        let s0 = (1.0 * 1.0 + 0.0 * 0.0) * scale;
        let s1 = (1.0 * 0.0 + 0.0 * 1.0) * scale;
        let e0 = s0.exp();
        let e1 = s1.exp();
        let sum_e = e0 + e1;
        let p0 = e0 / sum_e;
        let p1 = e1 / sum_e;
        let out0 = p0 * 1.0 + p1 * 0.0;
        let out1 = p0 * 0.0 + p1 * 1.0;

        assert!((result.get::<f32>(&[0, 0]).unwrap() - out0).abs() < 1e-5);
        assert!((result.get::<f32>(&[0, 1]).unwrap() - out1).abs() < 1e-5);
    }

    #[test]
    fn test_attention_two_queries() {
        // 2 queries, 1 head, head_dim=2, cache_len=2
        let q = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let k = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let v = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap();

        let result = attention(&q, &k, &v, 1, 1, 2).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        // Query 0 is [1,0], query 1 is [0,1]
        // Query 0 should attend more to key 0 ([1,0]) -> output closer to V0
        // Query 1 should attend more to key 1 ([0,1]) -> output closer to V1
        let out00: f32 = result.get(&[0, 0]).unwrap();
        let out01: f32 = result.get(&[0, 1]).unwrap();
        let out10: f32 = result.get(&[1, 0]).unwrap();
        let out11: f32 = result.get(&[1, 1]).unwrap();

        assert!(
            out00 > out10,
            "q0 should attend more to v0 ({out00} <= {out10})"
        );
        assert!(
            out11 > out01,
            "q1 should attend more to v1 ({out11} <= {out01})"
        );
    }

    // -----------------------------------------------------------------------
    // Tests: ContextError
    // -----------------------------------------------------------------------

    #[test]
    fn test_context_length_exceeded() {
        let model = load_minimal_model();
        let mut ctx = InferenceContext::new(model).unwrap();

        // Fill past context
        ctx.n_past = ctx.config().max_position_embeddings;

        // Try to process one more token
        let err = ctx.forward(&[0]).unwrap_err();
        assert!(matches!(
            err,
            ContextError::ContextLengthExceeded { max, requested }
            if max == ctx.config().max_position_embeddings && requested == max + 1
        ));
    }

    #[test]
    fn test_empty_tokens() {
        let model = load_minimal_model();
        let mut ctx = InferenceContext::new(model).unwrap();
        let result = ctx.forward(&[]).unwrap();
        assert_eq!(result.n_past, 0);
    }

    // -----------------------------------------------------------------------
    // Tests: KV cache
    // -----------------------------------------------------------------------

    #[test]
    fn test_kv_cache_update_and_retrieval() {
        let model = load_minimal_model();
        let mut ctx = InferenceContext::new(model).unwrap();

        // Override config for small test
        ctx.config = ModelConfig {
            architecture: "test".to_string(),
            vocab_size: 10,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_kv_heads: 1,
            max_position_embeddings: 10,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            bos_token_id: 1,
            eos_token_id: 2,
        };

        // Rebuild cache for our tiny config
        let max_seq_len = ctx.config.max_position_embeddings;
        let n_kv_heads = ctx.config.num_kv_heads;
        let head_dim = ctx.config.head_dim();
        let cache_shape = [max_seq_len, n_kv_heads * head_dim];
        ctx.k_cache = vec![Tensor::zeros(&cache_shape, DType::F32)];
        ctx.v_cache = vec![Tensor::zeros(&cache_shape, DType::F32)];

        // Manually set cache values for position 0
        let k_vals: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let v_vals: Vec<f32> = vec![0.5, 0.6, 0.7, 0.8];
        for h in 0..4 {
            ctx.k_cache[0].set(&[0, h], k_vals[h]).unwrap();
            ctx.v_cache[0].set(&[0, h], v_vals[h]).unwrap();
        }
        ctx.n_past = 1;

        // Retrieve via slice
        let k_view = ctx.k_cache[0].slice(&[0..1, 0..4]).unwrap();
        let v_view = ctx.v_cache[0].slice(&[0..1, 0..4]).unwrap();

        for h in 0..4 {
            let kv: f32 = k_view.get(&[0, h]).unwrap();
            assert!((kv - k_vals[h]).abs() < 1e-6);
            let vv: f32 = v_view.get(&[0, h]).unwrap();
            assert!((vv - v_vals[h]).abs() < 1e-6);
        }
    }

    // -----------------------------------------------------------------------
    // Tests: WeightNotFound
    // -----------------------------------------------------------------------

    #[test]
    fn test_weight_not_found() {
        let model = load_minimal_model();
        let ctx = InferenceContext::new(model).unwrap();
        let err = ctx.get_weight("nonexistent.weight").unwrap_err();
        assert!(matches!(err, ContextError::WeightNotFound(name) if name == "nonexistent.weight"));
    }

    // -----------------------------------------------------------------------
    // Tests: forward pass (end-to-end with fake weights)
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_creates_logits() {
        // Build a GGUF model with the minimal required tensors and config
        let arch = "llama";
        let kv_pairs: Vec<(&str, (u32, Vec<u8>))> = vec![
            ("llama.vocab_size", (10u32, u64_bytes(4))),
            ("llama.embedding_length", (10u32, u64_bytes(4))),
            ("llama.feed_forward_length", (10u32, u64_bytes(8))),
            ("llama.block_count", (10u32, u64_bytes(1))),
            ("llama.attention.head_count", (10u32, u64_bytes(2))),
            ("llama.attention.head_count_kv", (10u32, u64_bytes(2))),
            ("llama.context_length", (10u32, u64_bytes(64))),
            (
                "llama.attention.layer_norm_rms_epsilon",
                (6u32, f32_bytes(1e-5)),
            ),
            ("llama.rope.freq_base", (6u32, f32_bytes(10000.0))),
            ("tokenizer.ggml.bos_id", (5u32, i32_bytes(1))),
            ("tokenizer.ggml.eos_id", (5u32, i32_bytes(2))),
        ];

        // Build tensor data
        // token_embd.weight: [4, 4] (vocab_size=4, hidden=4) — identity
        // blk.0.attn_norm.weight: [4] — all 1s
        // blk.0.attn_q.weight: [4, 4]
        // blk.0.attn_k.weight: [4, 4]
        // blk.0.attn_v.weight: [4, 4]
        // blk.0.attn_output.weight: [4, 4]
        // blk.0.ffn_norm.weight: [4] — all 1s
        // blk.0.ffn_gate.weight: [8, 4]
        // blk.0.ffn_up.weight: [8, 4]
        // blk.0.ffn_down.weight: [4, 8]
        // output_norm.weight: [4] — all 1s
        // output.weight: [4, 4] (vocab=4, hidden=4)

        // We'll create a GGUF with all these tensors
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());

        // Count tensors (excluding the arch KV which we add separately)
        let tensor_names: Vec<&str> = vec![
            "token_embd.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "output_norm.weight",
            "output.weight",
        ];
        let tensor_count = tensor_names.len();
        let kv_count = 1 + kv_pairs.len(); // +1 for architecture

        buf.extend_from_slice(&(tensor_count as u64).to_le_bytes());
        buf.extend_from_slice(&(kv_count as u64).to_le_bytes());

        // Architecture KV
        let arch_key_bytes = b"general.architecture";
        let arch_val_bytes = arch.as_bytes();
        buf.extend_from_slice(&(arch_key_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(arch_key_bytes);
        buf.extend_from_slice(&8u32.to_le_bytes()); // String type
        buf.extend_from_slice(&(arch_val_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(arch_val_bytes);

        // Other KVs
        for (key, (ty, val_bytes)) in &kv_pairs {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&ty.to_le_bytes());
            buf.extend_from_slice(val_bytes);
        }

        // Tensor info entries and track their offset positions for patching
        // Shape format for GGUF: [innermost, ..., outermost]
        struct TensorEntry {
            name: &'static str,
            #[allow(dead_code)]
            gguf_dims: Vec<u64>, // reversed from standard shape
            num_elements: usize,
            offset_pos: usize,
        }

        let mut entries: Vec<TensorEntry> = Vec::new();
        let tensor_infos: Vec<(&str, &[usize])> = vec![
            // (name, standard_shape: [outermost, ..., innermost])
            ("token_embd.weight", &[4, 4]),
            ("blk.0.attn_norm.weight", &[4]),
            ("blk.0.attn_q.weight", &[4, 4]),
            ("blk.0.attn_k.weight", &[4, 4]),
            ("blk.0.attn_v.weight", &[4, 4]),
            ("blk.0.attn_output.weight", &[4, 4]),
            ("blk.0.ffn_norm.weight", &[4]),
            ("blk.0.ffn_gate.weight", &[8, 4]),
            ("blk.0.ffn_up.weight", &[8, 4]),
            ("blk.0.ffn_down.weight", &[4, 8]),
            ("output_norm.weight", &[4]),
            ("output.weight", &[4, 4]),
        ];

        for &(name, std_shape) in &tensor_infos {
            let gguf_dims: Vec<u64> = std_shape.iter().rev().map(|&d| d as u64).collect();
            let num_elements: usize = std_shape.iter().product();

            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&(std_shape.len() as u32).to_le_bytes());
            for &d in &gguf_dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&0u32.to_le_bytes()); // F32

            let offset_pos = buf.len();
            buf.extend_from_slice(&0u64.to_le_bytes()); // placeholder offset

            entries.push(TensorEntry {
                name,
                gguf_dims,
                num_elements,
                offset_pos,
            });
        }

        // Calculate data offsets (aligned)
        let infos_end = buf.len();
        let mut current_offset = align_up(infos_end as u64, GGUF_ALIGNMENT);

        // Patch offsets and later fill data
        let mut tensor_data: Vec<(&str, Vec<u8>)> = Vec::new();

        for entry in &entries {
            // Patch offset
            buf[entry.offset_pos..entry.offset_pos + 8]
                .copy_from_slice(&current_offset.to_le_bytes());

            // Create data: fill with sequential values based on position
            let size_bytes = entry.num_elements * 4; // f32 = 4 bytes
            let mut data = Vec::with_capacity(size_bytes);
            for i in 0..entry.num_elements {
                data.extend_from_slice(&(i as f32).to_le_bytes());
            }
            tensor_data.push((entry.name, data));

            current_offset += size_bytes as u64;
            current_offset = align_up(current_offset, GGUF_ALIGNMENT);
        }

        // Pad to first data offset
        let first_offset = entries[0].offset_pos;
        let expected_data_start = u64_from_le_slice(&buf[first_offset..first_offset + 8]);
        while buf.len() < expected_data_start as usize {
            buf.push(0);
        }

        // Write tensor data
        for (_name, data) in &tensor_data {
            buf.extend_from_slice(data);
            // Pad to alignment
            while buf.len() % GGUF_ALIGNMENT as usize != 0 {
                buf.push(0);
            }
        }

        let path = write_temp_gguf(&buf);
        let model = Model::load(&path).unwrap();
        let mut ctx = InferenceContext::new(model).unwrap();

        // Test forward with a single token
        let result = ctx.forward(&[0]).unwrap();
        assert_eq!(result.n_past, 1);
        assert_eq!(result.logits.shape(), &[4]); // vocab_size = 4

        // Process a second token
        let result2 = ctx.forward(&[1]).unwrap();
        assert_eq!(result2.n_past, 2);
        assert_eq!(result2.logits.shape(), &[4]);

        std::fs::remove_file(path).unwrap();
    }

    fn u64_from_le_slice(buf: &[u8]) -> u64 {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&buf[..8]);
        u64::from_le_bytes(arr)
    }
}
