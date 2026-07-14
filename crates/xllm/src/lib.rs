// SPDX-FileCopyrightText: 2026 XLLM contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

pub use xllm_bitnet as bitnet;
pub use xllm_context as context;
pub use xllm_ggml as ggml;
pub use xllm_model as model;
pub use xllm_quantize as quantize;
pub use xllm_sampling as sampling;
pub use xllm_tensor as tensor;
pub use xllm_tokenizer as tokenizer;
pub use xllm_train as train;

#[cfg(test)]
mod tests {
    #[test]
    fn re_exports_compile() {
        use core::mem::size_of;
        let _ = size_of::<crate::tensor::Tensor>();
        let _ = size_of::<crate::ggml::Graph>();
        let _ = size_of::<crate::model::Model>();
        let _ = size_of::<crate::tokenizer::Tokenizer>();
        let _ = size_of::<crate::sampling::Sampler>();
        let _ = size_of::<crate::context::InferenceContext>();
        let _ = size_of::<crate::quantize::Quantizer>();
        let _ = size_of::<crate::bitnet::BitNetKernel>();
        let _ = size_of::<crate::train::Trainer>();
    }
}
