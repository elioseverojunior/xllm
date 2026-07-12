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
        let _ = crate::tensor::Tensor;
        let _ = crate::ggml::Graph;
        let _ = crate::model::Model;
        let _ = crate::tokenizer::Tokenizer;
        let _ = crate::sampling::Sampler;
        let _ = crate::context::InferenceContext;
        let _ = crate::quantize::Quantizer;
        let _ = crate::bitnet::BitNetKernel;
        let _ = crate::train::Trainer;
    }
}
