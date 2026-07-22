// SPDX-FileCopyrightText: 2026 XLLM contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::HashMap, path::Path};

use half::f16;
use xllm_quantize::Quantizer;
use xllm_tensor::{DType, Tensor, TensorError};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];
const GGUF_VERSION: u32 = 3;
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;

#[cfg(test)]
const GGUF_ALIGNMENT: u64 = 32;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("invalid GGUF magic bytes")]
    InvalidMagic,
    #[error("unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    #[error("unexpected end of file")]
    UnexpectedEof,
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("tensor error: {0}")]
    TensorError(#[from] TensorError),
    #[error("tensor '{0}' not found")]
    TensorNotFound(String),
    #[error("unsupported tensor type: {0}")]
    UnsupportedTensorType(u32),
    #[error("invalid value type: {0}")]
    InvalidValueType(u32),
    #[error("numeric cast overflow: {0}")]
    CastOverflow(&'static str),
}

pub type Result<T> = std::result::Result<T, ModelError>;

// ---------------------------------------------------------------------------
// GGUFValueType (internal)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GGUFValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GGUFValueType {
    fn from_u32(v: u32) -> Option<Self> {
        LOOKUP.get(v as usize).copied().flatten()
    }
}

const LOOKUP: [Option<GGUFValueType>; 13] = [
    Some(GGUFValueType::Uint8),
    Some(GGUFValueType::Int8),
    Some(GGUFValueType::Uint16),
    Some(GGUFValueType::Int16),
    Some(GGUFValueType::Uint32),
    Some(GGUFValueType::Int32),
    Some(GGUFValueType::Float32),
    Some(GGUFValueType::Bool),
    Some(GGUFValueType::String),
    Some(GGUFValueType::Array),
    Some(GGUFValueType::Uint64),
    Some(GGUFValueType::Int64),
    Some(GGUFValueType::Float64),
];

// ---------------------------------------------------------------------------
// GGUFValue
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum GGUFValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Self>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

// ---------------------------------------------------------------------------
// TensorInfo
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub offset: u64,
    pub raw_type: u32,
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct Model {
    metadata: HashMap<String, GGUFValue>,
    tensors: HashMap<String, TensorInfo>,
    tensor_names: Vec<String>,
    data: Vec<u8>,
}

impl Model {
    /// Loads a GGUF v3 model file from the given path.
    ///
    /// Opens the file, reads it into memory, and parses the header, KV
    /// metadata, and tensor information sections.
    ///
    /// # Errors
    ///
    /// Returns `InvalidMagic` if the file does not start with the GGUF magic
    /// bytes. Returns `UnsupportedVersion` if the version is not 3. Returns
    /// `Io` for filesystem errors. Returns `UnexpectedEof` if the file is
    /// truncated.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let data = std::fs::read(path.as_ref())?;

        // Header: magic(4) + version(4) + tensor_count(8) + kv_count(8) = 24 bytes
        if data.len() < 24 {
            return Err(ModelError::UnexpectedEof);
        }

        let magic = &data[0..4];
        if magic != GGUF_MAGIC {
            return Err(ModelError::InvalidMagic);
        }

        let version = u32_from_le_slice(&data[4..8]);
        if version != GGUF_VERSION {
            return Err(ModelError::UnsupportedVersion(version));
        }

        let tensor_count = u64_from_le_slice(&data[8..16]);
        let kv_count = u64_from_le_slice(&data[16..24]);

        let mut pos: usize = 24;

        // --- KV pairs ---
        let mut metadata = HashMap::new();
        for _ in 0..kv_count {
            let (key, value, bytes_read) = Self::parse_kv(&data, pos)?;
            metadata.insert(key, value);
            pos += bytes_read;
        }

        // --- Tensor infos ---
        let mut tensors = HashMap::new();
        let mut tensor_names = Vec::with_capacity(usize::try_from(tensor_count).unwrap_or(0));
        for _ in 0..tensor_count {
            let (info, bytes_read) = Self::parse_tensor_info(&data, pos)?;
            tensor_names.push(info.name.clone());
            tensors.insert(info.name.clone(), info);
            pos += bytes_read;
        }

        tensor_names.sort();

        Ok(Self {
            metadata,
            tensors,
            tensor_names,
            data,
        })
    }

    // -----------------------------------------------------------------------
    // Metadata accessors
    // -----------------------------------------------------------------------

    /// Returns a reference to the full metadata map.
    #[must_use]
    pub const fn metadata(&self) -> &HashMap<String, GGUFValue> {
        &self.metadata
    }

    /// Looks up a single metadata value by key.
    #[must_use]
    pub fn metadata_value(&self, key: &str) -> Option<&GGUFValue> {
        self.metadata.get(key)
    }

    // -----------------------------------------------------------------------
    // Tensor info accessors
    // -----------------------------------------------------------------------

    /// Returns metadata for a named tensor.
    #[must_use]
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Returns all tensor names (sorted alphabetically).
    #[must_use]
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensor_names.clone()
    }

    /// Returns the number of tensors in the model.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    // -----------------------------------------------------------------------
    // Tensor data loading
    // -----------------------------------------------------------------------

    /// Loads tensor data from the memory-mapped file by name.
    ///
    /// # Errors
    ///
    /// Returns `TensorNotFound` if no tensor with the given name exists.
    /// Returns `TensorError` if the tensor data is malformed.
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| ModelError::TensorNotFound(name.to_string()))?;
        self.load_tensor(info)
    }

    /// Loads tensor data from the memory-mapped file by alphabetical index.
    ///
    /// # Errors
    ///
    /// Returns `TensorNotFound` if the index is out of range.
    /// Returns `TensorError` if the tensor data is malformed.
    pub fn tensor_by_index(&self, index: usize) -> Result<Tensor> {
        let name = self
            .tensor_names
            .get(index)
            .ok_or_else(|| ModelError::TensorNotFound(format!("index {index}")))?;
        self.tensor(name)
    }

    // -----------------------------------------------------------------------
    // Configuration helpers
    // -----------------------------------------------------------------------

    /// Returns the model architecture name (from `general.architecture`).
    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        self.metadata_value("general.architecture")
            .and_then(|v| match v {
                GGUFValue::String(s) => Some(s.as_str()),
                _ => None,
            })
    }

    /// Returns the maximum context length.
    #[must_use]
    pub fn context_length(&self) -> Option<u64> {
        self.arch_metadata("context_length")
    }

    /// Returns the embedding/hidden size.
    #[must_use]
    pub fn embedding_length(&self) -> Option<u64> {
        self.arch_metadata("embedding_length")
    }

    /// Returns the number of transformer blocks.
    #[must_use]
    pub fn block_count(&self) -> Option<u64> {
        self.arch_metadata("block_count")
    }

    /// Returns the number of attention heads.
    #[must_use]
    pub fn head_count(&self) -> Option<u64> {
        self.arch_metadata("attention.head_count")
    }

    /// Returns the number of key-value attention heads.
    #[must_use]
    pub fn head_count_kv(&self) -> Option<u64> {
        self.arch_metadata("attention.head_count_kv")
    }

    /// Returns the feed-forward intermediate size.
    #[must_use]
    pub fn feed_forward_length(&self) -> Option<u64> {
        self.arch_metadata("feed_forward_length")
    }

    /// Returns the `RoPE` base frequency.
    #[must_use]
    pub fn rope_freq_base(&self) -> Option<f32> {
        self.arch_metadata_f32("rope.freq_base")
    }

    /// Returns the RMS norm epsilon.
    #[must_use]
    pub fn rms_norm_eps(&self) -> Option<f32> {
        self.arch_metadata_f32("attn_norm_epsilon")
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Looks up metadata with a key formed as `{architecture}.{suffix}`.
    fn arch_metadata(&self, suffix: &str) -> Option<u64> {
        let arch = self.architecture()?;
        let key = format!("{arch}.{suffix}");
        // Try u32 first, then u64, then i32
        self.metadata_value(&key).and_then(|v| match *v {
            GGUFValue::Uint64(n) => Some(n),
            GGUFValue::Uint32(n) => Some(u64::from(n)),
            GGUFValue::Int32(n) if n >= 0 => u64::try_from(n).ok(),
            GGUFValue::Int64(n) if n >= 0 => u64::try_from(n).ok(),
            _ => None,
        })
    }

    /// Looks up a f32 metadata value with key `{architecture}.{suffix}`.
    fn arch_metadata_f32(&self, suffix: &str) -> Option<f32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.{suffix}");
        self.metadata_value(&key).and_then(|v| match *v {
            GGUFValue::Float32(n) => Some(n),
            GGUFValue::Float64(n) => Some(truncate_f64_to_f32(n)),
            _ => None,
        })
    }

    /// Parses a KV pair at the given position.
    fn parse_kv(data: &[u8], pos: usize) -> Result<(String, GGUFValue, usize)> {
        let (key, key_len) = parse_gguf_string(data, pos)?;
        let mut offset = pos + key_len;

        if offset + 4 > data.len() {
            return Err(ModelError::UnexpectedEof);
        }
        let type_raw = u32_from_le_slice(&data[offset..offset + 4]);
        offset += 4;

        let value_type =
            GGUFValueType::from_u32(type_raw).ok_or(ModelError::InvalidValueType(type_raw))?;

        let (value, value_len) = Self::parse_value(data, offset, value_type)?;

        Ok((key, value, key_len + 4 + value_len))
    }

    /// Parses a typed value at the given position.
    fn parse_value(data: &[u8], pos: usize, ty: GGUFValueType) -> Result<(GGUFValue, usize)> {
        macro_rules! read_int {
            ($ty:ident, $variant:ident, $reader:ident, $size:expr) => {{
                if pos + $size > data.len() {
                    return Err(ModelError::UnexpectedEof);
                }
                let v = $reader(&data[pos..pos + $size]);
                Ok((GGUFValue::$variant(v), $size))
            }};
        }
        match ty {
            GGUFValueType::Uint8 => {
                if pos + 1 > data.len() {
                    return Err(ModelError::UnexpectedEof);
                }
                Ok((GGUFValue::Uint8(data[pos]), 1))
            }
            GGUFValueType::Int8 => {
                if pos + 1 > data.len() {
                    return Err(ModelError::UnexpectedEof);
                }
                Ok((GGUFValue::Int8(i8::from_ne_bytes([data[pos]])), 1))
            }
            GGUFValueType::Uint16 => read_int!(u16, Uint16, u16_from_le_slice, 2),
            GGUFValueType::Int16 => read_int!(i16, Int16, i16_from_le_slice, 2),
            GGUFValueType::Uint32 => read_int!(u32, Uint32, u32_from_le_slice, 4),
            GGUFValueType::Int32 => read_int!(i32, Int32, i32_from_le_slice, 4),
            GGUFValueType::Float32 => read_int!(f32, Float32, f32_from_le_slice, 4),
            GGUFValueType::Bool => {
                if pos + 1 > data.len() {
                    return Err(ModelError::UnexpectedEof);
                }
                Ok((GGUFValue::Bool(data[pos] != 0), 1))
            }
            GGUFValueType::String => {
                let (s, n) = parse_gguf_string(data, pos)?;
                Ok((GGUFValue::String(s), n))
            }
            GGUFValueType::Array => Self::parse_array(data, pos),
            GGUFValueType::Uint64 => read_int!(u64, Uint64, u64_from_le_slice, 8),
            GGUFValueType::Int64 => read_int!(i64, Int64, i64_from_le_slice, 8),
            GGUFValueType::Float64 => read_int!(f64, Float64, f64_from_le_slice, 8),
        }
    }

    fn parse_array(data: &[u8], pos: usize) -> Result<(GGUFValue, usize)> {
        if pos + 12 > data.len() {
            return Err(ModelError::UnexpectedEof);
        }
        let item_type_raw = u32_from_le_slice(&data[pos..pos + 4]);
        let array_len = u64_from_le_slice(&data[pos + 4..pos + 12]);
        let item_type = GGUFValueType::from_u32(item_type_raw)
            .ok_or(ModelError::InvalidValueType(item_type_raw))?;
        let mut offset = pos + 12;
        let mut items = Vec::with_capacity(usize::try_from(array_len).unwrap_or(0));
        for _ in 0..array_len {
            let (val, n) = Self::parse_value(data, offset, item_type)?;
            items.push(val);
            offset += n;
        }
        Ok((GGUFValue::Array(items), offset - pos))
    }

    /// Parses a tensor info entry at the given position.
    fn parse_tensor_info(data: &[u8], pos: usize) -> Result<(TensorInfo, usize)> {
        let (name, name_len) = parse_gguf_string(data, pos)?;
        let mut offset = pos + name_len;

        // n_dims
        if offset + 4 > data.len() {
            return Err(ModelError::UnexpectedEof);
        }
        let n_dims = u32_from_le_slice(&data[offset..offset + 4]) as usize;
        offset += 4;

        // dims (reversed in GGUF: innermost first)
        let dims_byte_len = n_dims * 8;
        if offset + dims_byte_len > data.len() {
            return Err(ModelError::UnexpectedEof);
        }
        let mut dims = Vec::with_capacity(n_dims);
        for i in 0..n_dims {
            let d = u64_from_le_slice(&data[offset + i * 8..offset + (i + 1) * 8]);
            dims.push(
                usize::try_from(d).map_err(|_| ModelError::CastOverflow("tensor dimension"))?,
            );
        }
        offset += dims_byte_len;

        // tensor_type
        if offset + 4 > data.len() {
            return Err(ModelError::UnexpectedEof);
        }
        let tensor_type = u32_from_le_slice(&data[offset..offset + 4]);
        offset += 4;

        // offset in file
        if offset + 8 > data.len() {
            return Err(ModelError::UnexpectedEof);
        }
        let file_offset = u64_from_le_slice(&data[offset..offset + 8]);
        offset += 8;

        // Convert GGUF reversed dims to standard order (outermost first)
        let mut shape: Vec<usize> = dims.iter().rev().copied().collect();
        if shape.is_empty() {
            shape.push(1);
        }

        let (dt, raw_type) = match tensor_type {
            GGML_TYPE_F32 => (DType::F32, GGML_TYPE_F32),
            GGML_TYPE_F16 => (DType::F16, GGML_TYPE_F16),
            GGML_TYPE_Q4_0 => (DType::F32, GGML_TYPE_Q4_0),
            _ => return Err(ModelError::UnsupportedTensorType(tensor_type)),
        };

        Ok((
            TensorInfo {
                name,
                shape,
                dtype: dt,
                offset: file_offset,
                raw_type,
            },
            offset - pos,
        ))
    }

    /// Loads tensor data from in-memory buffer given a `TensorInfo`.
    fn load_tensor(&self, info: &TensorInfo) -> Result<Tensor> {
        let start =
            usize::try_from(info.offset).map_err(|_| ModelError::CastOverflow("tensor offset"))?;
        let num_elem: usize = info.shape.iter().product();

        if info.raw_type == GGML_TYPE_Q4_0 {
            let num_blocks = num_elem.div_ceil(32);
            let byte_size = num_blocks * 18;
            if start + byte_size > self.data.len() {
                return Err(ModelError::UnexpectedEof);
            }
            let bytes = &self.data[start..start + byte_size];
            let tensor =
                Quantizer::dequantize_q4_0(bytes, &info.shape).map_err(ModelError::TensorError)?;
            Ok(tensor)
        } else {
            let elem_size = info.dtype.byte_size();
            let byte_size = num_elem * elem_size;
            if start + byte_size > self.data.len() {
                return Err(ModelError::UnexpectedEof);
            }
            let bytes = &self.data[start..start + byte_size];
            match info.dtype {
                DType::F32 => {
                    let mut values = Vec::with_capacity(num_elem);
                    for chunk in bytes.as_chunks::<4>().0 {
                        values.push(f32::from_le_bytes(*chunk));
                    }
                    Ok(Tensor::from_slice(&values, &info.shape)?)
                }
                DType::F16 => {
                    let mut values = Vec::with_capacity(num_elem);
                    for chunk in bytes.as_chunks::<2>().0 {
                        values.push(f16::from_le_bytes(*chunk));
                    }
                    Ok(Tensor::from_slice(&values, &info.shape)?)
                }
            }
        }
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
#[must_use]
const fn truncate_f64_to_f32(v: f64) -> f32 {
    v as f32
}

fn u32_from_le_slice(buf: &[u8]) -> u32 {
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&buf[..4]);
    u32::from_le_bytes(arr)
}

fn u64_from_le_slice(buf: &[u8]) -> u64 {
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&buf[..8]);
    u64::from_le_bytes(arr)
}

fn i32_from_le_slice(buf: &[u8]) -> i32 {
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&buf[..4]);
    i32::from_le_bytes(arr)
}

fn i16_from_le_slice(buf: &[u8]) -> i16 {
    let mut arr = [0u8; 2];
    arr.copy_from_slice(&buf[..2]);
    i16::from_le_bytes(arr)
}

fn i64_from_le_slice(buf: &[u8]) -> i64 {
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&buf[..8]);
    i64::from_le_bytes(arr)
}

fn f32_from_le_slice(buf: &[u8]) -> f32 {
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&buf[..4]);
    f32::from_le_bytes(arr)
}

fn f64_from_le_slice(buf: &[u8]) -> f64 {
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&buf[..8]);
    f64::from_le_bytes(arr)
}

fn u16_from_le_slice(buf: &[u8]) -> u16 {
    let mut arr = [0u8; 2];
    arr.copy_from_slice(&buf[..2]);
    u16::from_le_bytes(arr)
}

/// Parses a `GGUFString` at the given position. Returns (string, bytes consumed).
fn parse_gguf_string(data: &[u8], pos: usize) -> Result<(String, usize)> {
    if pos + 8 > data.len() {
        return Err(ModelError::UnexpectedEof);
    }
    let len = usize::try_from(u64_from_le_slice(&data[pos..pos + 8]))
        .map_err(|_| ModelError::CastOverflow("string length"))?;
    let start = pos + 8;
    let end = start + len;
    if end > data.len() {
        return Err(ModelError::UnexpectedEof);
    }
    let s = String::from_utf8_lossy(&data[start..end]).to_string();
    Ok((s, 8 + len))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::approx_constant,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown
)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn align_up(val: u64, alignment: u64) -> u64 {
        (val + alignment - 1) & !(alignment - 1)
    }

    fn create_test_gguf(tensor_data: &[f32]) -> Vec<u8> {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(b"GGUF"); // magic
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&1u64.to_le_bytes()); // kv_count

        // KV pair: test_key = 42 (Int32)
        let key = b"test_key";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&5u32.to_le_bytes()); // Int32 type
        buf.extend_from_slice(&42i32.to_le_bytes());

        // Tensor info: "test_tensor", GGUF shape [4, 3] (innermost, outermost)
        let tensor_name = b"test_tensor";
        buf.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        buf.extend_from_slice(tensor_name);
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&4u64.to_le_bytes()); // dims[0] = innermost
        buf.extend_from_slice(&3u64.to_le_bytes()); // dims[1] = outermost
        buf.extend_from_slice(&0u32.to_le_bytes()); // tensor_type = F32

        // Offset: align current position + 8 (for offset field) to 32 bytes
        let data_offset_hint = buf.len() + 8;
        let aligned_offset = align_up(data_offset_hint as u64, GGUF_ALIGNMENT);
        buf.extend_from_slice(&aligned_offset.to_le_bytes());

        // Pad to aligned offset
        while buf.len() < aligned_offset as usize {
            buf.push(0);
        }

        // Tensor data
        for &v in tensor_data {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        buf
    }

    /// Writes GGUF bytes to a temp file and returns the path.
    fn write_temp_gguf(data: &[u8]) -> std::path::PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "test_{}_{}.gguf",
            std::process::id(),
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::write(&path, data).unwrap();
        path
    }

    /// Creates a GGUF binary with multiple value types for testing.
    fn create_typed_value_gguf() -> Vec<u8> {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&5u64.to_le_bytes()); // kv_count

        // KV 1: string_val = "hello"
        let key1 = b"string_key";
        let val1 = b"hello";
        buf.extend_from_slice(&(key1.len() as u64).to_le_bytes());
        buf.extend_from_slice(key1);
        buf.extend_from_slice(&8u32.to_le_bytes()); // String type
        buf.extend_from_slice(&(val1.len() as u64).to_le_bytes());
        buf.extend_from_slice(val1);

        // KV 2: float_val = 3.14
        let key2 = b"float_key";
        buf.extend_from_slice(&(key2.len() as u64).to_le_bytes());
        buf.extend_from_slice(key2);
        buf.extend_from_slice(&6u32.to_le_bytes()); // Float32 type
        buf.extend_from_slice(&3.14f32.to_le_bytes());

        // KV 3: bool_val = true
        let key3 = b"bool_key";
        buf.extend_from_slice(&(key3.len() as u64).to_le_bytes());
        buf.extend_from_slice(key3);
        buf.extend_from_slice(&7u32.to_le_bytes()); // Bool type
        buf.push(1u8);

        // KV 4: array_val = [1, 2, 3] (Int32 values)
        let key4 = b"array_key";
        buf.extend_from_slice(&(key4.len() as u64).to_le_bytes());
        buf.extend_from_slice(key4);
        buf.extend_from_slice(&9u32.to_le_bytes()); // Array type
        buf.extend_from_slice(&5u32.to_le_bytes()); // item type = Int32
        buf.extend_from_slice(&3u64.to_le_bytes()); // array length
        buf.extend_from_slice(&1i32.to_le_bytes());
        buf.extend_from_slice(&2i32.to_le_bytes());
        buf.extend_from_slice(&3i32.to_le_bytes());

        // KV 5: uint64_val = 18446744073709551615
        let key5 = b"uint64_key";
        buf.extend_from_slice(&(key5.len() as u64).to_le_bytes());
        buf.extend_from_slice(key5);
        buf.extend_from_slice(&10u32.to_le_bytes()); // Uint64 type
        buf.extend_from_slice(&u64::MAX.to_le_bytes());

        buf
    }

    // -----------------------------------------------------------------------
    // Tests: header validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_load_valid_header() {
        let gguf = create_test_gguf(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]);
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        assert_eq!(model.tensor_count(), 1);
        assert_eq!(model.metadata().len(), 1);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_invalid_magic() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"XXXX");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let path = write_temp_gguf(&buf);
        let err = Model::load(&path).unwrap_err();
        assert!(matches!(err, ModelError::InvalidMagic));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_unsupported_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&2u32.to_le_bytes()); // version 2
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let path = write_temp_gguf(&buf);
        let err = Model::load(&path).unwrap_err();
        assert!(matches!(err, ModelError::UnsupportedVersion(2)));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_truncated_file() {
        let buf = vec![0x47, 0x47, 0x55, 0x46]; // just magic, nothing else
        let path = write_temp_gguf(&buf);
        let err = Model::load(&path).unwrap_err();
        assert!(matches!(err, ModelError::UnexpectedEof));
        std::fs::remove_file(path).unwrap();
    }

    // -----------------------------------------------------------------------
    // Tests: KV metadata parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_metadata_kv_int32() {
        let gguf = create_test_gguf(&[1.0; 12]);
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let val = model.metadata_value("test_key").unwrap();
        assert_eq!(*val, GGUFValue::Int32(42));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_metadata_kv_string() {
        let gguf = create_typed_value_gguf();
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let val = model.metadata_value("string_key").unwrap();
        assert_eq!(*val, GGUFValue::String("hello".to_string()));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_metadata_kv_float32() {
        let gguf = create_typed_value_gguf();
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let val = model.metadata_value("float_key").unwrap();
        assert!(matches!(val, GGUFValue::Float32(f) if (*f - 3.14).abs() < 1e-6));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_metadata_kv_bool() {
        let gguf = create_typed_value_gguf();
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let val = model.metadata_value("bool_key").unwrap();
        assert_eq!(*val, GGUFValue::Bool(true));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_metadata_kv_array() {
        let gguf = create_typed_value_gguf();
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let val = model.metadata_value("array_key").unwrap();
        assert_eq!(
            *val,
            GGUFValue::Array(vec![
                GGUFValue::Int32(1),
                GGUFValue::Int32(2),
                GGUFValue::Int32(3),
            ])
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_metadata_kv_uint64_max() {
        let gguf = create_typed_value_gguf();
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let val = model.metadata_value("uint64_key").unwrap();
        assert_eq!(*val, GGUFValue::Uint64(u64::MAX));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_metadata_not_found() {
        let gguf = create_test_gguf(&[1.0; 12]);
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        assert!(model.metadata_value("nonexistent").is_none());
        std::fs::remove_file(path).unwrap();
    }

    // -----------------------------------------------------------------------
    // Tests: tensor info parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_tensor_info_shape_reversal() {
        // GGUF stores dims as [innermost, outermost] = [4, 3]
        // Standard shape should be [3, 4]
        let gguf = create_test_gguf(&[1.0; 12]);
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let info = model.tensor_info("test_tensor").unwrap();
        assert_eq!(info.shape, vec![3, 4]);
        assert_eq!(info.dtype, DType::F32);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_tensor_info_not_found() {
        let gguf = create_test_gguf(&[1.0; 12]);
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        assert!(model.tensor_info("nonexistent").is_none());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_tensor_names() {
        let mut buf = Vec::new();
        // Header: 2 tensors, 0 kv
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes()); // 2 tensors
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 KV

        // -- All tensor infos first --
        // Tensor "b_tensor": shape [2,2], GGUF dims [2,2]
        let name_b = b"b_tensor";
        buf.extend_from_slice(&(name_b.len() as u64).to_le_bytes());
        buf.extend_from_slice(name_b);
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        let t1_off_pos = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes()); // placeholder offset

        // Tensor "a_tensor": shape [2,2], GGUF dims [2,2]
        let name_a = b"a_tensor";
        buf.extend_from_slice(&(name_a.len() as u64).to_le_bytes());
        buf.extend_from_slice(name_a);
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        let t2_off_pos = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes()); // placeholder offset

        let infos_end = buf.len();

        // Calculate data offsets (must be >= infos_end and 32-byte aligned)
        let data_size: u64 = 4 * 4; // 4 floats per tensor * 4 bytes = 16 bytes
        let t1_offset = align_up(infos_end as u64, GGUF_ALIGNMENT);
        let t2_offset = align_up(t1_offset + data_size, GGUF_ALIGNMENT);

        // Patch offsets
        buf[t1_off_pos..t1_off_pos + 8].copy_from_slice(&t1_offset.to_le_bytes());
        buf[t2_off_pos..t2_off_pos + 8].copy_from_slice(&t2_offset.to_le_bytes());

        // Pad to t1_offset
        while buf.len() < t1_offset as usize {
            buf.push(0);
        }
        // Tensor b_tensor data (4 floats)
        for _ in 0..4 {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }

        // Pad to t2_offset
        while buf.len() < t2_offset as usize {
            buf.push(0);
        }
        // Tensor a_tensor data (4 floats)
        for _ in 0..4 {
            buf.extend_from_slice(&2.0f32.to_le_bytes());
        }

        let path = write_temp_gguf(&buf);
        let model = Model::load(&path).unwrap();

        let names = model.tensor_names();
        assert_eq!(names, vec!["a_tensor", "b_tensor"]);
        std::fs::remove_file(path).unwrap();
    }

    // -----------------------------------------------------------------------
    // Tests: tensor data loading
    // -----------------------------------------------------------------------

    #[test]
    fn test_tensor_f32_data() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let gguf = create_test_gguf(&data);
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();

        let tensor = model.tensor("test_tensor").unwrap();
        assert_eq!(tensor.shape(), &[3, 4]);
        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(tensor.size(), 12);

        // Verify data: standard shape [3, 4], row-major
        // GGUF dims [4, 3] -> standard [3, 4]
        for i in 0..3 {
            for j in 0..4 {
                let expected = (i * 4 + j) as f32;
                let val: f32 = tensor.get(&[i, j]).unwrap();
                assert!((val - expected).abs() < 1e-6, "mismatch at [{i}, {j}]");
            }
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_tensor_by_index() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let gguf = create_test_gguf(&data);
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();

        let tensor = model.tensor_by_index(0).unwrap();
        assert_eq!(tensor.shape(), &[3, 4]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_tensor_not_found() {
        let gguf = create_test_gguf(&[1.0; 12]);
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let err = model.tensor("nonexistent").unwrap_err();
        assert!(matches!(err, ModelError::TensorNotFound(_)));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_tensor_by_index_out_of_range() {
        let gguf = create_test_gguf(&[1.0; 12]);
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let err = model.tensor_by_index(99).unwrap_err();
        assert!(matches!(err, ModelError::TensorNotFound(_)));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_tensor_1d() {
        let mut buf = Vec::new();

        // Header: 1 tensor, 0 KV
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 KV

        // Tensor info: 1D, shape [5]
        let name = b"vec";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        buf.extend_from_slice(&5u64.to_le_bytes()); // dims[0] = innermost = 5
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32

        let data_offset_hint = (buf.len() + 8) as u64;
        let aligned = (data_offset_hint + GGUF_ALIGNMENT - 1) & !(GGUF_ALIGNMENT - 1);
        buf.extend_from_slice(&aligned.to_le_bytes());

        while buf.len() < aligned as usize {
            buf.push(0);
        }

        // Data: [10, 20, 30, 40, 50]
        buf.extend_from_slice(&10.0f32.to_le_bytes());
        buf.extend_from_slice(&20.0f32.to_le_bytes());
        buf.extend_from_slice(&30.0f32.to_le_bytes());
        buf.extend_from_slice(&40.0f32.to_le_bytes());
        buf.extend_from_slice(&50.0f32.to_le_bytes());

        let path = write_temp_gguf(&buf);
        let model = Model::load(&path).unwrap();
        let info = model.tensor_info("vec").unwrap();
        assert_eq!(info.shape, &[5]);

        let tensor = model.tensor("vec").unwrap();
        assert_eq!(tensor.shape(), &[5]);
        assert!((tensor.get::<f32>(&[0]).unwrap() - 10.0).abs() < 1e-6);
        assert!((tensor.get::<f32>(&[4]).unwrap() - 50.0).abs() < 1e-6);

        std::fs::remove_file(path).unwrap();
    }

    // -----------------------------------------------------------------------
    // Tests: zero KV pairs
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_kv_pairs() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 KV

        let path = write_temp_gguf(&buf);
        let model = Model::load(&path).unwrap();
        assert_eq!(model.metadata().len(), 0);
        assert_eq!(model.tensor_count(), 0);
        std::fs::remove_file(path).unwrap();
    }

    // -----------------------------------------------------------------------
    // Tests: configuration helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_architecture() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());

        // KV: general.architecture = "llama"
        let key = b"general.architecture";
        let val = b"llama";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // String type
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val);

        let path = write_temp_gguf(&buf);
        let model = Model::load(&path).unwrap();
        assert_eq!(model.architecture(), Some("llama"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_architecture_not_found() {
        let buf = create_test_gguf(&[1.0; 12]);
        let path = write_temp_gguf(&buf);
        let model = Model::load(&path).unwrap();
        assert!(model.architecture().is_none());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_context_length() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());

        // KV 1: general.architecture = "llama"
        let k1 = b"general.architecture";
        let v1 = b"llama";
        buf.extend_from_slice(&(k1.len() as u64).to_le_bytes());
        buf.extend_from_slice(k1);
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&(v1.len() as u64).to_le_bytes());
        buf.extend_from_slice(v1);

        // KV 2: llama.context_length = 4096
        let k2 = b"llama.context_length";
        buf.extend_from_slice(&(k2.len() as u64).to_le_bytes());
        buf.extend_from_slice(k2);
        buf.extend_from_slice(&10u32.to_le_bytes()); // Uint64 type
        buf.extend_from_slice(&4096u64.to_le_bytes());

        let path = write_temp_gguf(&buf);
        let model = Model::load(&path).unwrap();
        assert_eq!(model.context_length(), Some(4096));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_context_length_no_arch() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());

        // KV: llama.context_length = 4096 but no architecture
        let k = b"llama.context_length";
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k);
        buf.extend_from_slice(&10u32.to_le_bytes());
        buf.extend_from_slice(&4096u64.to_le_bytes());

        let path = write_temp_gguf(&buf);
        let model = Model::load(&path).unwrap();
        // No architecture means no prefix lookup
        assert!(model.context_length().is_none());
        std::fs::remove_file(path).unwrap();
    }

    // -----------------------------------------------------------------------
    // Tests: unsupported tensor type
    // -----------------------------------------------------------------------

    #[test]
    fn test_unsupported_tensor_type() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        let name = b"bad_tensor";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&99u32.to_le_bytes()); // invalid tensor type
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        let path = write_temp_gguf(&buf);
        let err = Model::load(&path).unwrap_err();
        assert!(matches!(err, ModelError::UnsupportedTensorType(99)));
        std::fs::remove_file(path).unwrap();
    }

    // -----------------------------------------------------------------------
    // Tests: round-trip binary parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_round_trip_binary() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let gguf_bytes = create_test_gguf(&data);
        let path = write_temp_gguf(&gguf_bytes);
        let model = Model::load(&path).unwrap();

        // Verify header
        assert_eq!(model.tensor_count(), 1);

        // Verify KV
        let kv = model.metadata_value("test_key").unwrap();
        assert_eq!(*kv, GGUFValue::Int32(42));

        // Verify tensor info
        let info = model.tensor_info("test_tensor").unwrap();
        assert_eq!(info.shape, &[3, 4]);
        assert_eq!(info.dtype, DType::F32);

        // Verify tensor data
        let tensor = model.tensor("test_tensor").unwrap();
        assert_eq!(tensor.shape(), &[3, 4]);
        for (i, &expected) in data.iter().enumerate() {
            let row = i / 4;
            let col = i % 4;
            let val: f32 = tensor.get(&[row, col]).unwrap();
            assert!((val - expected).abs() < 1e-6);
        }

        std::fs::remove_file(path).unwrap();
    }

    // -----------------------------------------------------------------------
    // Tests: invalid value type
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_value_type() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());

        let key = b"bad_type";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&255u32.to_le_bytes()); // invalid type

        let path = write_temp_gguf(&buf);
        let err = Model::load(&path).unwrap_err();
        assert!(matches!(err, ModelError::InvalidValueType(255)));
        std::fs::remove_file(path).unwrap();
    }

    // -----------------------------------------------------------------------
    // Tests: multiple tensors
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_tensors() {
        let mut buf = Vec::new();

        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes()); // 2 tensors
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 KV

        // -- All tensor infos first, with offsets past all infos --
        // Tensor 1 info: "first", GGUF shape [3, 2] (innermost, outermost)
        let n1 = b"first";
        buf.extend_from_slice(&(n1.len() as u64).to_le_bytes());
        buf.extend_from_slice(n1);
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&3u64.to_le_bytes()); // dims[0] = innermost
        buf.extend_from_slice(&2u64.to_le_bytes()); // dims[1] = outermost
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        // offset placeholder (8 bytes) — will patch later
        let t1_off_pos = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes());

        // Tensor 2 info: "second", GGUF shape [4, 1] (innermost, outermost)
        let n2 = b"second";
        buf.extend_from_slice(&(n2.len() as u64).to_le_bytes());
        buf.extend_from_slice(n2);
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&4u64.to_le_bytes()); // dims[0] = innermost
        buf.extend_from_slice(&1u64.to_le_bytes()); // dims[1] = outermost
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        // offset placeholder (8 bytes) — will patch later
        let t2_off_pos = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes());

        let infos_end = buf.len();

        // Calculate data offsets (must be >= infos_end and 32-byte aligned)
        let t1_offset = align_up(infos_end as u64, GGUF_ALIGNMENT);
        let t1_data_size: u64 = 6 * 4; // 6 floats, 24 bytes
        let t2_offset = align_up(t1_offset + t1_data_size, GGUF_ALIGNMENT);

        // Patch the offsets
        buf[t1_off_pos..t1_off_pos + 8].copy_from_slice(&t1_offset.to_le_bytes());
        buf[t2_off_pos..t2_off_pos + 8].copy_from_slice(&t2_offset.to_le_bytes());

        // Pad to t1_offset
        while buf.len() < t1_offset as usize {
            buf.push(0);
        }

        // Tensor 1 data: [0, 1, 2, 3, 4, 5] (shape [2,3]: 2 rows, 3 cols)
        for i in 0..6 {
            buf.extend_from_slice(&(i as f32).to_le_bytes());
        }

        // Pad to t2_offset
        while buf.len() < t2_offset as usize {
            buf.push(0);
        }

        // Tensor 2 data: [10, 20, 30, 40] (shape [1,4]: 1 row, 4 cols)
        buf.extend_from_slice(&10.0f32.to_le_bytes());
        buf.extend_from_slice(&20.0f32.to_le_bytes());
        buf.extend_from_slice(&30.0f32.to_le_bytes());
        buf.extend_from_slice(&40.0f32.to_le_bytes());

        let path = write_temp_gguf(&buf);
        let model = Model::load(&path).unwrap();

        assert_eq!(model.tensor_count(), 2);
        assert_eq!(model.tensor_names().len(), 2);

        let t1 = model.tensor("first").unwrap();
        assert_eq!(t1.shape(), &[2, 3]); // standard: outermost first
        assert!((t1.get::<f32>(&[0, 0]).unwrap() - 0.0).abs() < 1e-6);
        assert!((t1.get::<f32>(&[1, 2]).unwrap() - 5.0).abs() < 1e-6);

        let t2 = model.tensor("second").unwrap();
        assert_eq!(t2.shape(), &[1, 4]);
        assert!((t2.get::<f32>(&[0, 0]).unwrap() - 10.0).abs() < 1e-6);
        assert!((t2.get::<f32>(&[0, 3]).unwrap() - 40.0).abs() < 1e-6);

        std::fs::remove_file(path).unwrap();
    }

    // -----------------------------------------------------------------------
    // Tests: F16 tensor type
    // -----------------------------------------------------------------------

    #[test]
    fn test_tensor_f16_data() {
        let mut buf = Vec::new();

        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        // Tensor: "f16_tensor", 1D shape [4]
        let name = b"f16_tensor";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        buf.extend_from_slice(&4u64.to_le_bytes()); // dims[0] = 4
        buf.extend_from_slice(&1u32.to_le_bytes()); // tensor_type = F16

        let off_hint = (buf.len() + 8) as u64;
        let offset = (off_hint + GGUF_ALIGNMENT - 1) & !(GGUF_ALIGNMENT - 1);
        buf.extend_from_slice(&offset.to_le_bytes());

        while buf.len() < offset as usize {
            buf.push(0);
        }

        // F16 data: [1.0, 2.0, 3.0, 4.0]
        let vals = [1.0f32, 2.0, 3.0, 4.0];
        for &v in &vals {
            let h = f16::from_f32(v);
            buf.extend_from_slice(&h.to_le_bytes());
        }

        let path = write_temp_gguf(&buf);
        let model = Model::load(&path).unwrap();

        let info = model.tensor_info("f16_tensor").unwrap();
        assert_eq!(info.dtype, DType::F16);
        assert_eq!(info.shape, &[4]);

        let tensor = model.tensor("f16_tensor").unwrap();
        assert_eq!(tensor.dtype(), DType::F16);

        for (i, &expected) in vals.iter().enumerate() {
            let val: f16 = tensor.get(&[i]).unwrap();
            assert!((val.to_f32() - expected).abs() < 1e-3);
        }

        std::fs::remove_file(path).unwrap();
    }

    // -----------------------------------------------------------------------
    // Tests: Q4_0 tensor type
    // -----------------------------------------------------------------------

    #[allow(clippy::cast_sign_loss)]
    #[test]
    fn test_tensor_q4_0_data() {
        // Q4_0 block: 1 f16 scale + 16 bytes nibbles = 18 bytes per 32 values
        // We create 32 values: [0..31] as f32, quantize to Q4_0 format
        let vals: Vec<f32> = (0..32).map(|i| i as f32).collect();

        // Compute Q4_0 data: d = max_abs/7, nibble = clamp(round(val/d + 8), 0, 15)
        let max_abs = 31.0f32;
        let d = max_abs / 7.0;
        let d_bits = f16::from_f32(d).to_bits();

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 KV pairs

        // Tensor: "q4_0_tensor", 1D shape [32]
        let name = b"q4_0_tensor";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        buf.extend_from_slice(&32u64.to_le_bytes()); // dims[0] = 32
        buf.extend_from_slice(&2u32.to_le_bytes()); // tensor_type = Q4_0

        let off_hint = (buf.len() + 8) as u64;
        let offset = (off_hint + GGUF_ALIGNMENT - 1) & !(GGUF_ALIGNMENT - 1);
        buf.extend_from_slice(&offset.to_le_bytes());

        while buf.len() < offset as usize {
            buf.push(0);
        }

        // Q4_0 block data: d (f16) + 16 bytes nibbles
        buf.extend_from_slice(&d_bits.to_le_bytes());
        for (i, &v) in vals.iter().enumerate() {
            let q = ((v / d + 8.0).round() as i32).clamp(0, 15) as u8;
            let byte_idx = i / 2;
            if byte_idx >= buf.len() - offset as usize - 2 {
                buf.push(0);
            }
            if i & 1 == 0 {
                buf[offset as usize + 2 + byte_idx] =
                    (buf[offset as usize + 2 + byte_idx] & 0xf0) | q;
            } else {
                buf[offset as usize + 2 + byte_idx] =
                    (buf[offset as usize + 2 + byte_idx] & 0x0f) | (q << 4);
            }
        }

        let path = write_temp_gguf(&buf);
        let model = Model::load(&path).unwrap();

        let info = model.tensor_info("q4_0_tensor").unwrap();
        assert_eq!(info.dtype, DType::F32); // logical type after dequant
        assert_eq!(info.shape, &[32]);
        assert_eq!(info.raw_type, 2);

        let tensor = model.tensor("q4_0_tensor").unwrap();
        assert_eq!(tensor.dtype(), DType::F32);

        // Verify dequantized values are close to original
        for (i, &expected) in vals.iter().enumerate() {
            let val: f32 = tensor.get(&[i]).unwrap();
            assert!(
                (val - expected).abs() < d + 0.1,
                "mismatch at {i}: deq={val}, expected={expected}, d={d}"
            );
        }

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_tensor_q4_0_short_data() {
        // Minimal GGUF with Q4_0 tensor but truncated data
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        let name = b"short_tensor";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&100u64.to_le_bytes()); // 100 elements = 4 blocks = 72 bytes
        buf.extend_from_slice(&2u32.to_le_bytes()); // Q4_0

        buf.extend_from_slice(&0u64.to_le_bytes()); // offset = 0

        let path = write_temp_gguf(&buf);
        let model = Model::load(&path).unwrap();
        let err = model.tensor("short_tensor").unwrap_err();
        assert!(matches!(err, ModelError::UnexpectedEof));

        std::fs::remove_file(path).unwrap();
    }
}
