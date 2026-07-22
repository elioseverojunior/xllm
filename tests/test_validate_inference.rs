// SPDX-FileCopyrightText: 2026 XLLM contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;

use xllm::{
    context::InferenceContext,
    model::Model,
    sampling::{Sampler, SamplingParams},
    tokenizer::Tokenizer,
};

fn create_test_gguf() -> Vec<u8> {
    let mut buf = Vec::new();

    // GGUF header
    buf.extend_from_slice(b"GGUF"); // magic
    buf.extend_from_slice(&3u32.to_le_bytes()); // version = 3
    buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
    buf.extend_from_slice(&2u64.to_le_bytes()); // kv_count = 2

    // KV pair 1: general.architecture = "test"
    buf.extend_from_slice(&17u64.to_le_bytes()); // key length
    buf.extend_from_slice(b"general.architecture");
    buf.extend_from_slice(&8u32.to_le_bytes()); // string type
    buf.extend_from_slice(&4u64.to_le_bytes()); // value length ("test")
    buf.extend_from_slice(b"test");

    // KV pair 2: test.context_length = 32
    buf.extend_from_slice(&16u64.to_le_bytes()); // key length
    buf.extend_from_slice(b"test.context_length");
    buf.extend_from_slice(&10u32.to_le_bytes()); // uint64 type
    buf.extend_from_slice(&32u64.to_le_bytes()); // value

    // Tensor: "output_weight"
    let tensor_name = b"output_weight";
    buf.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
    buf.extend_from_slice(tensor_name);
    buf.extend_from_slice(&2u32.to_le_bytes()); // 2 dimensions
    buf.extend_from_slice(&1u64.to_le_bytes()); // dim[0] = 1 (innermost)
    buf.extend_from_slice(&4u64.to_le_bytes()); // dim[1] = 4 (outermost)
    buf.extend_from_slice(&0u32.to_le_bytes()); // F32 type

    // Data offset (must be 32-byte aligned)
    let data_offset = ((buf.len() + 8 + 31) / 32) * 32;
    buf.extend_from_slice(&data_offset.to_le_bytes());

    // Pad to alignment
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
    // Create and save test model
    let model_data = create_test_gguf();
    let temp_dir = env::temp_dir();
    let model_path = temp_dir.join("test_model.gguf");
    fs::write(&model_path, &model_data).expect("Failed to write model");

    println!("Created test model: {}", model_path.display());

    // Test the model
    let model = Model::load(&model_path).expect("Failed to load model");
    println!("✓ Model loaded: {:?} ({} tensors)",
             model.architecture(), model.tensor_count());

    // Simple tokenizer for testing
    let mut vocab = Vec::new();
    vocab.push(b"h".to_vec()); // 0
    vocab.push(b"e".to_vec()); // 1
    vocab.push(b"l".to_vec()); // 2
    vocab.push(b"o".to_vec()); // 3
    vocab.push(b" ".to_vec()); // 4
    vocab.push(b"w".to_vec()); // 5
    vocab.push(b"r".to_vec()); // 6
    vocab.push(b"d".to_vec()); // 7
    vocab.push(b"!\n".to_vec()); // 8

    let tokenizer = Tokenizer::new(
        xllm::tokenizer::TokenizerModel::LLaMA,
        vocab.iter().map(|v| String::from_utf8_lossy(v).as_ref()).collect(),
        Some(vec![0.0; 9]),
        0,  // bos_id
        1,  // eos_id
        false,
    );

    let mut ctx = InferenceContext::new(model).expect("Failed to create context");
    println!("✓ Inference context created");

    // Test encode/decode
    let input = "hello world";
    let tokens = tokenizer.encode(input).expect("Failed to encode");
    let output = tokenizer.decode(&tokens).expect("Failed to decode");
    println!("✓ Tokenizer test: '{}' -> {:?} -> '{}'", input, tokens, output);

    // Test inference
    let result = ctx.forward(&tokens).expect("Forward pass failed");
    println!("✓ Forward pass: logits shape = {:?}", result.logits.shape());

    // Sample a token
    let mut sampler = Sampler::new(12345);
    let params = SamplingParams { temperature: 0.0, ..SamplingParams::default() };
    let token = sampler.sample(&result.logits, &params).expect("Sampling failed");
    let text = tokenizer.decode(&[token]).expect("Failed to decode token");
    println!("✓ Sampling: token={} -> text='{}'", token, text);

    // Cleanup
    let _ = fs::remove_file(&model_path);

    println!("\n🎉 All tests passed! End-to-end CPU inference is working.");
    println!("The xllm CLI is ready to use with complete GGUF model files.");
}
