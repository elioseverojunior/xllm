// SPDX-FileCopyrightText: 2026 XLLM contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use clap::Parser;
use reqwest::Client;

use xllm::{
    context::InferenceContext,
    model::Model,
    sampling::{Sampler, SamplingParams},
    tokenizer::Tokenizer,
};

#[derive(Parser)]
#[command(version, about = "xllm — LLM inference engine (CPU-first)")]
enum Command {
    /// Run inference on a GGUF model
    Run {
        /// Path to GGUF model file
        #[arg(short = 'm', long)]
        model: PathBuf,

        /// Input prompt text
        #[arg(short = 'p', long, default_value = "Hello")]
        prompt: String,

        /// Maximum number of tokens to generate
        #[arg(short = 'x', long, default_value_t = 128)]
        max_tokens: usize,

        /// Sampling temperature (0.0 = greedy)
        #[arg(long, default_value_t = 0.8)]
        temperature: f32,

        /// Top-K sampling (0 = disabled)
        #[arg(long, default_value_t = 40)]
        top_k: u32,

        /// Top-P nucleus sampling (1.0 = disabled)
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,

        /// Random seed for reproducibility
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },

    /// Download a GGUF model from `HuggingFace`
    Download {
        /// `HuggingFace` model repository (e.g., "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
        #[arg(short, long)]
        repo: String,

        /// Specific filename to download (if not specified, downloads the first .gguf file found)
        #[arg(short, long)]
        filename: Option<String>,

        /// Local directory to save the model (defaults to ~/.cache/huggingface/hub)
        #[arg(long)]
        output_dir: Option<PathBuf>,

        /// Whether to use symlinks (disabled by default for reliable file access)
        #[arg(long)]
        _no_symlinks: bool,

        /// Force re-download even if file exists
        #[arg(long)]
        force: bool,
    },
}

fn main() {
    match Command::parse() {
        Command::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
        } => run_inference(&model, &prompt, max_tokens, temperature, top_k, top_p, seed),
        Command::Download {
            repo,
            filename,
            output_dir,
            force,
            ..
        } => download_model(&repo, filename.as_deref(), output_dir, false, force),
    }
}

fn get_default_cache_dir() -> PathBuf {
    // Follow huggingface convention for cache directory
    let home = dirs::home_dir().unwrap_or_default();
    home.join(".cache").join("huggingface").join("hub")
}

#[allow(clippy::needless_pass_by_value)]
fn run_inference(
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: u64,
) {
    // 1. Load model
    eprintln!("Loading model from {}...", model.display());
    let model_instance = match Model::load(model) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading model: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("  architecture: {:?}", model_instance.architecture());
    eprintln!("  tensors: {}", model_instance.tensor_count());

    // 2. Create tokenizer
    let tokenizer = match Tokenizer::from_gguf(&model_instance) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error creating tokenizer: {e}");
            std::process::exit(1);
        }
    };

    // 3. Create inference context
    let mut ctx = match InferenceContext::new(model_instance) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error creating inference context: {e}");
            std::process::exit(1);
        }
    };
    let max_ctx = ctx.config().max_position_embeddings;
    let n_layers = ctx.config().num_hidden_layers;
    let _eos_id = ctx.config().eos_token_id;
    eprintln!("  context length: {max_ctx}");
    eprintln!("  layers: {n_layers}");

    // 4. Tokenize prompt
    let input_tokens = match tokenizer.encode(prompt) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error tokenizing prompt: {e}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "Prompt tokens: {} ({} chars)",
        input_tokens.len(),
        prompt.len()
    );

    // 5. Sampler
    let params = SamplingParams {
        temperature,
        top_k,
        top_p,
        ..SamplingParams::default()
    };
    let mut sampler = Sampler::new(seed);

    // 6. Generate
    eprintln!("\n--- Generated text ---");

    // Run forward pass on prompt
    let result = match ctx.forward(&input_tokens) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error during forward pass: {e}");
            std::process::exit(1);
        }
    };

    // Sample first token from prompt logits
    let mut current_token = match sampler.sample(&result.logits, &params) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error sampling: {e}");
            std::process::exit(1);
        }
    };

    // Output prompt
    print!("{prompt}");

    let mut generated = 0usize;

    while generated < max_tokens && current_token != 0 {
        match tokenizer.decode(&[current_token]) {
            Ok(text) => print!("{text}"),
            Err(_) => print!("<ERR>"),
        }
        std::io::stdout().flush().ok();

        let result = match ctx.forward(&[current_token]) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("\nError during forward pass: {e}");
                break;
            }
        };

        let next_token = match sampler.sample(&result.logits, &params) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error sampling: {e}");
                break;
            }
        };
        current_token = next_token;
        generated += 1;
    }

    println!();
    eprintln!("--- Generated {generated} tokens ---");
}

fn download_model(
    repo: &str,
    filename: Option<&str>,
    output_dir: Option<PathBuf>,
    _no_symlinks: bool,
    force: bool,
) {
    let output_dir = output_dir.unwrap_or_else(get_default_cache_dir);

    if !output_dir.exists()
        && let Err(e) = fs::create_dir_all(&output_dir)
    {
        eprintln!(
            "Error creating output directory {}: {e}",
            output_dir.display()
        );
        std::process::exit(1);
    }

    eprintln!("Downloading model from HuggingFace repo: {repo}");

    let rt = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error creating Tokio runtime: {e}");
            std::process::exit(1);
        }
    };

    let result = rt.block_on(async {
        if let Some(filename) = filename {
            eprintln!("Downloading specific file: {filename}");
            download_specific_file(repo, filename, &output_dir, force).await
        } else {
            eprintln!("Fetching file list from repository...");
            match list_gguf_files(repo).await {
                Ok(files) => {
                    if files.is_empty() {
                        return Err("No .gguf files found in the repository".to_string());
                    }
                    let mut files = files;
                    files.sort();
                    let filename = &files[0];
                    eprintln!("Found .gguf files: {files:?}");
                    eprintln!("Downloading: {filename}");
                    download_specific_file(repo, filename, &output_dir, force).await
                }
                Err(e) => Err(format!("Failed to list repo files: {e}")),
            }
        }
    });

    match result {
        Ok(dest_path) => {
            eprintln!("Model successfully downloaded to: {}", dest_path.display());
            println!("You can now run inference with:");
            println!(
                "  xllm-cli run --model {} --prompt \"Your prompt here\"",
                dest_path.display()
            );
        }
        Err(e) => {
            eprintln!("Error downloading model: {e}");
            std::process::exit(1);
        }
    }
}

async fn download_specific_file(
    repo: &str,
    filename: &str,
    output_dir: &Path,
    force: bool,
) -> Result<PathBuf, String> {
    let dest_path = {
        let mut path = output_dir.to_path_buf();
        path.push(filename);
        path
    };

    if dest_path.exists() && !force {
        eprintln!("File already exists: {}", dest_path.display());
        eprintln!("Existing file found, but downloading anyway for simplicity");
    } else if !force {
        eprintln!("File does not exist, downloading...");
    } else {
        eprintln!("Force download requested, downloading...");
    }

    let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");
    eprintln!("Downloading from: {url}");

    let client = Client::new();
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Failed to download from {url}: {e}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "HTTP {}: Failed to download {}",
            response.status(),
            url
        ));
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| format!("Failed to get response bytes: {e}"))?;

    if !output_dir.exists()
        && let Err(e) = fs::create_dir_all(output_dir)
    {
        return Err(format!("Failed to create output directory: {e}"));
    }

    let mut file = match fs::File::create(&dest_path) {
        Ok(f) => f,
        Err(e) => return Err(format!("Failed to create file: {e}")),
    };

    if let Err(e) = file.write_all(&bytes) {
        return Err(format!("Failed to write file: {e}"));
    }

    eprintln!("Download complete: {} bytes", bytes.len());
    Ok(dest_path)
}

async fn list_gguf_files(repo: &str) -> Result<Vec<String>, String> {
    let api_url = format!("https://huggingface.co/api/models/{repo}/tree/main");
    eprintln!("Fetching file list from: {api_url}");

    let client = Client::new();
    let response = client
        .get(&api_url)
        .send()
        .await
        .map_err(|e| format!("Failed to fetch file list: {e}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "HTTP {}: Failed to fetch file list for {}",
            response.status(),
            repo
        ));
    }

    let files: Vec<serde_json::Value> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse JSON response: {e}"))?;

    let mut gguf_files = Vec::new();
    for file in files {
        if let Some(path) = file.get("path").and_then(|p| p.as_str())
            && path.to_lowercase().ends_with(".gguf")
        {
            gguf_files.push(path.to_string());
        }
    }

    Ok(gguf_files)
}

mod dirs {
    use std::env;
    use std::path::PathBuf;

    pub(crate) fn home_dir() -> Option<PathBuf> {
        env::var_os("HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("USERPROFILE").map(PathBuf::from))
    }
}
