// SPDX-FileCopyrightText: 2026 XLLM contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Creates a tiny test GGUF model for quick validation
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use std::process;

use xllm::{
    context::InferenceContext,
    model::Model,
    sampling::{Sampler, SamplingParams},
    tokenizer::Tokenizer,
};

fn create_test_model() -> Vec<u8> {
    let mut buf = Vec::new();

    // GGUF header
    buf.extend_from_slice(b"GGUF"); // magic
    buf.extend_from_slice(&3u32.to_le_bytes()); // version
    buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count: 1
    buf.extend_from_slice(&2u64.to_le_bytes()); // kv_count: 2

    // KV pairs
    // general.architecture = "test"
    buf.extend_from_slice(&17u64.to_le_bytes()); // key length
    buf.extend_from_slice(b"general.architecture");
    buf.extend_from_slice(&8u32.to_le_bytes()); // string type
    buf.extend_from_slice(&4u64.to_le_bytes()); // value length
    buf.extend_from_slice(b"test");

    // test.context_length = 32
    buf.extend_from_slice(&16u64.to_le_bytes()); // key length
    buf.extend_from_slice(b"test.context_length");
    buf.extend_from_slice(&10u32.to_le_bytes()); // uint64 type
    buf.extend_from_slice(&32u64.to_le_bytes()); // value

    // Tensor info: "output_weight", shape [1, 4], F32
    let tensor_name = b"output_weight";
    buf.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
    buf.extend_from_slice(tensor_name);
    buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
    buf.extend_from_slice(&1u64.to_le_bytes()); // dim[0] = 1 (innermost)
    buf.extend_from_slice(&4u64.to_le_bytes()); // dim[1] = 4 (outermost)
    buf.extend_from_slice(&0u32.to_le_bytes()); // F32 type

    // Data offset aligned to 32 bytes
    let data_offset = ((buf.len() + 8 + 31) / 32) * 32;
    buf.extend_from_slice(&data_offset.to_le_bytes());

    // Pad to data offset
    while buf.len() < data_offset as usize {
        buf.push(0);
    }

    // Tensor data: [1.0, 0.0, 0.0, 0.0]
    buf.extend_from_slice(&1.0f32.to_le_bytes());
    buf.extend_from_slice(&0.0f32.to_le_bytes());
    buf.extend_from_slice(&0.0f32.to_le_bytes());
    buf.extend_from_slice(&0.0f32.to_le_bytes());

    buf
}

fn main() {
    // Create test model
    let model_data = create_test_model();

    // Save to temp file
    let temp_dir = env::temp_dir();
    let model_path = temp_dir.join("test_model.gguf");
    fs::write(&model_path, &model_data).unwrap();

    println!("Testing with model: {}", model_path.display());

    // Load model
    let model = Model::load(&model_path).expect("Failed to load model");
    println!("Model loaded: architecture={:?}, tensors={}",
             model.architecture(), model.tensor_count());

    // Create simple tokenizer
    let mut vocab = Vec::new();
    vocab.push(b"h".to_vec());
    vocab.push(b"e".to_vec());
    vocab.push(b"l".to_vec());
    vocab.push(b"o".to_vec());
    vocab.push(b" ".to_vec());
    vocab.push(b"w".to_vec());
    vocab.push(b"r".to_vec());
    vocab.push(b"d".to_vec());
    vocab.push(b"!\n".to_vec());

    let tokenizer = Tokenizer::new(
        xllm::tokenizer::TokenizerModel::LLaMA,
        vocab.iter().map(|v| String::from_utf8_lossy(v).as_ref()).collect(),
        Some(vec![0.0; 9]),
        0,
        1,
        false,
    );

    // Create context
    let mut ctx = InferenceContext::new(model).expect("Failed to create context");

    // Process input
    let input = "hello";
    let tokens = tokenizer.encode(input).expect("Failed to tokenize");
    println!("Input '{}' -> tokens: {:?}", input, tokens);

    // Forward pass
    let result = ctx.forward(&tokens).expect("Forward failed");
    println!("Got logits shape: {:?}", result.logits.shape());

    // Sample token
    let mut sampler = Sampler::new(42);
    let params = SamplingParams { temperature: 0.0, ..SamplingParams::default() };
    let next_token = sampler.sample(&result.logits, &params).expect("Sampling failed");
    println!("Sampled token: {}", next_token);

    // Decode output
    let output = tokenizer.decode(&[next_token]).expect("Decode failed");
    println!("Output: '{}'", output);

    // Cleanup
    let _ = fs::remove_file(&model_path);

    println!("\n✅ End-to-end CPU inference test successful!");
}
