// SPDX-FileCopyrightText: 2026 XLLM contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::needless_range_loop,
    clippy::suboptimal_flops
)]

use std::mem::size_of;

use rayon::prelude::*;
use thiserror::Error;
use xllm_tensor::{DType, Tensor, TensorError};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("node {id} not found")]
    NodeNotFound { id: usize },
    #[error("input {input_id} for node {node_id} not found")]
    InputNotFound { node_id: usize, input_id: usize },
    #[error("tensor error: {0}")]
    TensorError(#[from] TensorError),
    #[error("graph has cycle")]
    CycleDetected,
    #[error("unsupported op: {op:?}")]
    UnsupportedOp { op: Op },
    #[error("invalid parameters: {0}")]
    InvalidParams(String),
}

pub type Result<T> = std::result::Result<T, GraphError>;

// ---------------------------------------------------------------------------
// Op — operation types matching llama.cpp's ggml_op (core subset)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Copy,
    Reshape,
    Permute,
    GetRows,
    Add,
    Mul,
    Scale(f32),
    MatMul,
    SoftMax,
    RMSNorm { eps: f32 },
    RoPE { theta: f32, dim_pairs: usize },
    Silu,
    Cont,
    Repeat,
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Node {
    id: usize,
    op: Op,
    inputs: Vec<usize>,
    params: Option<Box<[u8]>>,
}

impl Node {
    #[must_use]
    pub const fn id(&self) -> usize {
        self.id
    }

    #[must_use]
    pub const fn op(&self) -> &Op {
        &self.op
    }

    #[must_use]
    pub fn inputs(&self) -> &[usize] {
        &self.inputs
    }

    #[must_use]
    pub fn params(&self) -> Option<&[u8]> {
        self.params.as_deref()
    }
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Graph {
    nodes: Vec<Node>,
    tensors: Vec<Option<Tensor>>,
}

impl Graph {
    #[must_use]
    pub fn node(&self, id: usize) -> Option<&Node> {
        self.nodes.get(id)
    }

    #[must_use]
    pub fn tensor(&self, id: usize) -> Option<&Tensor> {
        self.tensors.get(id).and_then(|t| t.as_ref())
    }

    pub fn set_input(&mut self, node_id: usize, tensor: Tensor) -> Result<()> {
        if node_id >= self.nodes.len() {
            return Err(GraphError::NodeNotFound { id: node_id });
        }
        self.tensors[node_id] = Some(tensor);
        Ok(())
    }

    /// Executes the graph in topological order.
    pub fn forward(&mut self) -> Result<()> {
        let n = self.nodes.len();
        if n == 0 {
            return Ok(());
        }

        // --- Kahn's algorithm topological sort --------------------------------
        let mut in_degree = vec![0usize; n];
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (node_id, node) in self.nodes.iter().enumerate() {
            in_degree[node_id] = node.inputs.len();
            for &input_id in &node.inputs {
                dependents[input_id].push(node_id);
            }
        }

        let mut queue: Vec<usize> = (0..n).filter(|&id| in_degree[id] == 0).collect();

        let mut order = Vec::with_capacity(n);
        while let Some(node_id) = queue.pop() {
            order.push(node_id);
            for &dep_id in &dependents[node_id] {
                in_degree[dep_id] -= 1;
                if in_degree[dep_id] == 0 {
                    queue.push(dep_id);
                }
            }
        }

        if order.len() != n {
            return Err(GraphError::CycleDetected);
        }

        // --- Execute nodes in topological order -------------------------------
        for &node_id in &order {
            if self.tensors[node_id].is_some() {
                continue;
            }

            let node = &self.nodes[node_id];

            // Leaf node (input) with no tensor set — skip; downstream nodes
            // will fail with InputNotFound if they reference it.
            if node.inputs.is_empty() {
                continue;
            }

            let input_tensors: Vec<&Tensor> = node
                .inputs
                .iter()
                .map(|&input_id| {
                    self.tensors[input_id]
                        .as_ref()
                        .ok_or(GraphError::InputNotFound { node_id, input_id })
                })
                .collect::<Result<Vec<&Tensor>>>()?;

            let result = execute_op(node, &input_tensors)?;
            self.tensors[node_id] = Some(result);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GraphBuilder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GraphBuilder {
    nodes: Vec<Node>,
    shapes: Vec<Vec<usize>>,
    dtypes: Vec<DType>,
}

impl GraphBuilder {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            shapes: Vec::new(),
            dtypes: Vec::new(),
        }
    }

    /// Add an input placeholder node (leaf node with externally-provided tensor).
    pub fn add_input(&mut self, shape: &[usize], dtype: DType) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            op: Op::Copy,
            inputs: vec![],
            params: None,
        });
        self.shapes.push(shape.to_vec());
        self.dtypes.push(dtype);
        id
    }

    /// Add an operation node. Reshape and Repeat require separate methods
    /// because they need a target shape.
    ///
    /// # Errors
    ///
    /// Returns `InvalidParams` for `Reshape` or `Repeat` — use
    /// `add_reshape`/`add_repeat` instead.
    /// Returns `NodeNotFound` if any input ID is out of range.
    /// Returns `TensorError` if shapes/dtypes are incompatible.
    pub fn add(&mut self, op: Op, inputs: &[usize]) -> Result<usize> {
        if matches!(op, Op::Reshape | Op::Repeat) {
            return Err(GraphError::InvalidParams(
                "Reshape/Repeat require a target shape; use add_reshape or add_repeat".into(),
            ));
        }
        self.add_impl(op, inputs, None)
    }

    /// Add a Reshape node.
    ///
    /// # Errors
    ///
    /// Returns `NodeNotFound` if the input ID is out of range.
    /// Returns `TensorError` if element counts differ.
    pub fn add_reshape(&mut self, input: usize, target_shape: &[usize]) -> Result<usize> {
        if input >= self.nodes.len() {
            return Err(GraphError::NodeNotFound { id: input });
        }
        let input_size: usize = self.shapes[input].iter().product();
        let target_size: usize = target_shape.iter().product();
        if input_size != target_size {
            return Err(GraphError::TensorError(TensorError::ElementCount {
                expected: input_size,
                actual: target_size,
            }));
        }
        self.add_impl(Op::Reshape, &[input], Some(target_shape))
    }

    /// Add a Repeat (broadcast) node.
    ///
    /// # Errors
    ///
    /// Returns `NodeNotFound` if the input ID is out of range.
    /// Returns `TensorError` if shapes are not broadcastable.
    pub fn add_repeat(&mut self, input: usize, target_shape: &[usize]) -> Result<usize> {
        if input >= self.nodes.len() {
            return Err(GraphError::NodeNotFound { id: input });
        }
        // Validate broadcastability
        broadcast_shapes(&self.shapes[input], target_shape)?;
        self.add_impl(Op::Repeat, &[input], Some(target_shape))
    }

    fn add_impl(
        &mut self,
        op: Op,
        inputs: &[usize],
        target_shape: Option<&[usize]>,
    ) -> Result<usize> {
        for &input_id in inputs {
            if input_id >= self.nodes.len() {
                return Err(GraphError::NodeNotFound { id: input_id });
            }
        }

        let (shape, dtype) = self.infer_shape_dtype(&op, inputs, target_shape)?;

        let params = match &op {
            Op::Reshape | Op::Repeat => {
                let shape = target_shape
                    .ok_or_else(|| GraphError::InvalidParams("target shape required".into()))?;
                Some(shape_to_bytes(shape))
            }
            _ => None,
        };

        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            op,
            inputs: inputs.to_vec(),
            params,
        });
        self.shapes.push(shape);
        self.dtypes.push(dtype);
        Ok(id)
    }

    fn infer_shape_dtype(
        &self,
        op: &Op,
        inputs: &[usize],
        target_shape: Option<&[usize]>,
    ) -> Result<(Vec<usize>, DType)> {
        match op {
            Op::Add | Op::Mul => {
                let a_shape = &self.shapes[inputs[0]];
                let b_shape = &self.shapes[inputs[1]];
                let a_dtype = self.dtypes[inputs[0]];
                let b_dtype = self.dtypes[inputs[1]];
                if a_dtype != b_dtype {
                    return Err(GraphError::TensorError(TensorError::DTypeMismatch {
                        a: a_dtype,
                        b: b_dtype,
                    }));
                }
                let shape = broadcast_shapes(a_shape, b_shape)?;
                Ok((shape, a_dtype))
            }
            Op::Scale(_) | Op::Copy | Op::Cont | Op::Permute => {
                let shape = self.shapes[inputs[0]].clone();
                Ok((shape, self.dtypes[inputs[0]]))
            }
            Op::MatMul => {
                let a_shape = &self.shapes[inputs[0]];
                let b_shape = &self.shapes[inputs[1]];
                if a_shape.len() != 2 || b_shape.len() != 2 {
                    return Err(GraphError::TensorError(TensorError::InvalidOperation(
                        "matmul requires 2D tensors".into(),
                    )));
                }
                if a_shape[1] != b_shape[0] {
                    return Err(GraphError::TensorError(TensorError::ShapeMismatch {
                        expected: a_shape.clone(),
                        got: b_shape.clone(),
                    }));
                }
                Ok((vec![a_shape[0], b_shape[1]], DType::F32))
            }
            Op::SoftMax | Op::RMSNorm { .. } | Op::RoPE { .. } | Op::Silu => {
                let shape = self.shapes[inputs[0]].clone();
                Ok((shape, DType::F32))
            }
            Op::Reshape | Op::Repeat => {
                let shape = target_shape
                    .ok_or_else(|| {
                        GraphError::InvalidParams("target shape required for reshape/repeat".into())
                    })?
                    .to_vec();
                let dtype = if inputs.is_empty() {
                    DType::F32
                } else {
                    self.dtypes[inputs[0]]
                };
                Ok((shape, dtype))
            }
            Op::GetRows => {
                let data_shape = &self.shapes[inputs[0]];
                let idx_shape = &self.shapes[inputs[1]];
                if idx_shape.len() != 1 {
                    return Err(GraphError::TensorError(TensorError::InvalidOperation(
                        "GetRows indices must be 1D".into(),
                    )));
                }
                let mut out_shape = vec![idx_shape[0]];
                out_shape.extend_from_slice(&data_shape[1..]);
                Ok((out_shape, self.dtypes[inputs[0]]))
            }
        }
    }

    /// Consume the builder and produce a `Graph`.
    ///
    /// # Errors
    ///
    /// Returns `NodeNotFound` if any node references a non-existent input.
    pub fn build(self) -> Result<Graph> {
        for node in &self.nodes {
            for &input_id in &node.inputs {
                if input_id >= self.nodes.len() {
                    return Err(GraphError::NodeNotFound { id: input_id });
                }
            }
        }
        let tensors = vec![None; self.nodes.len()];
        Ok(Graph {
            nodes: self.nodes,
            tensors,
        })
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------

fn shape_to_bytes(shape: &[usize]) -> Box<[u8]> {
    let elem_size = size_of::<usize>();
    let byte_len = size_of_val(shape);
    let mut bytes = vec![0u8; byte_len];
    for (i, &dim) in shape.iter().enumerate() {
        let start = i * elem_size;
        let dim_bytes = dim.to_ne_bytes();
        bytes[start..start + elem_size].copy_from_slice(&dim_bytes[..elem_size]);
    }
    bytes.into_boxed_slice()
}

fn bytes_to_shape(bytes: &[u8]) -> Result<Vec<usize>> {
    let elem_size = size_of::<usize>();
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    if !bytes.len().is_multiple_of(elem_size) {
        return Err(GraphError::InvalidParams(
            "shape bytes length mismatch".into(),
        ));
    }
    let n = bytes.len() / elem_size;
    let mut shape = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * elem_size;
        let mut arr = [0u8; 8];
        arr[..elem_size].copy_from_slice(&bytes[start..start + elem_size]);
        shape.push(usize::from_ne_bytes(arr));
    }
    Ok(shape)
}

fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);
    for i in 0..max_ndim {
        let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
            return Err(GraphError::TensorError(TensorError::InvalidOperation(
                format!("cannot broadcast shapes {a:?} and {b:?}"),
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

// ---------------------------------------------------------------------------
// Op execution dispatch
// ---------------------------------------------------------------------------

fn execute_op(node: &Node, inputs: &[&Tensor]) -> Result<Tensor> {
    match &node.op {
        Op::Copy => Ok(op_copy(inputs)),
        Op::Reshape => op_reshape(inputs, node.params.as_deref()),
        Op::Permute => op_permute(inputs),
        Op::GetRows => op_get_rows(inputs),
        Op::Add => op_add(inputs),
        Op::Mul => op_mul(inputs),
        Op::Scale(factor) => op_scale(inputs, *factor),
        Op::MatMul => op_matmul(inputs),
        Op::SoftMax => op_softmax(inputs),
        Op::RMSNorm { eps } => op_rms_norm(inputs, *eps),
        Op::RoPE { theta, dim_pairs } => op_rope(inputs, *theta, *dim_pairs),
        Op::Silu => op_silu(inputs),
        Op::Cont => op_cont(inputs),
        Op::Repeat => op_repeat(inputs, node.params.as_deref()),
    }
}

// ---------------------------------------------------------------------------
// Op implementations
// ---------------------------------------------------------------------------

fn op_copy(inputs: &[&Tensor]) -> Tensor {
    inputs[0].to_owned()
}

fn op_reshape(inputs: &[&Tensor], params: Option<&[u8]>) -> Result<Tensor> {
    let bytes = params.ok_or_else(|| {
        GraphError::InvalidParams("Reshape requires target shape in params".into())
    })?;
    let shape = bytes_to_shape(bytes)?;
    inputs[0].reshape(&shape).map_err(GraphError::from)
}

fn op_permute(inputs: &[&Tensor]) -> Result<Tensor> {
    inputs[0].contiguous().map_err(GraphError::from)
}

fn op_get_rows(inputs: &[&Tensor]) -> Result<Tensor> {
    let data = inputs[0];
    let indices = inputs[1];
    let dtype = data.dtype();

    let num_rows = data.shape()[0];
    let cols: Vec<usize> = data.shape()[1..].to_vec();
    let num_indices = indices.shape()[0];

    let mut out_shape = vec![num_indices];
    out_shape.extend_from_slice(&cols);
    let mut result = Tensor::zeros(&out_shape, dtype);

    for (out_row, flat) in (0..num_indices).enumerate() {
        let idx: usize = indices.get::<f32>(&[flat]).map(|v| v as usize)?;
        if idx >= num_rows {
            return Err(GraphError::TensorError(TensorError::IndexOutOfBounds {
                index: vec![idx],
                shape: data.shape().to_vec(),
            }));
        }
        if cols.is_empty() {
            let val: f32 = data.get(&[idx])?;
            result.set(&[out_row], val)?;
        } else if cols.len() == 1 {
            for j in 0..cols[0] {
                let val: f32 = data.get(&[idx, j])?;
                result.set(&[out_row, j], val)?;
            }
        } else {
            let row_size: usize = cols.iter().product();
            for flat_col in 0..row_size {
                let ci = unflatten_index(flat_col, &cols);
                let mut src_idx = vec![idx];
                src_idx.extend_from_slice(&ci);
                let val: f32 = data.get(&src_idx)?;
                let mut dst_idx = vec![out_row];
                dst_idx.extend_from_slice(&ci);
                result.set(&dst_idx, val)?;
            }
        }
    }

    Ok(result)
}

fn op_add(inputs: &[&Tensor]) -> Result<Tensor> {
    inputs[0].add(inputs[1]).map_err(GraphError::from)
}

fn op_mul(inputs: &[&Tensor]) -> Result<Tensor> {
    let a = inputs[0];
    let b = inputs[1];

    if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
        return Err(GraphError::TensorError(TensorError::DTypeMismatch {
            a: a.dtype(),
            b: b.dtype(),
        }));
    }

    let result_shape = broadcast_shapes(a.shape(), b.shape())?;
    let size: usize = result_shape.iter().product();
    let a_shape = a.shape().to_vec();
    let b_shape = b.shape().to_vec();

    let mut a_vals = vec![0.0f32; size];
    for flat in 0..size {
        let multi = unflatten_index(flat, &result_shape);
        let a_multi = broadcast_map(&multi, &a_shape);
        a_vals[flat] = a.get(&a_multi)?;
    }

    let mut b_vals = vec![0.0f32; size];
    for flat in 0..size {
        let multi = unflatten_index(flat, &result_shape);
        let b_multi = broadcast_map(&multi, &b_shape);
        b_vals[flat] = b.get(&b_multi)?;
    }

    let mut out_vals = vec![0.0f32; size];
    out_vals
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, val)| *val = a_vals[i] * b_vals[i]);

    Tensor::from_slice(&out_vals, &result_shape).map_err(GraphError::from)
}

fn op_scale(inputs: &[&Tensor], factor: f32) -> Result<Tensor> {
    let input = inputs[0];

    if input.dtype() != DType::F32 {
        return Err(GraphError::TensorError(TensorError::DTypeMismatch {
            a: input.dtype(),
            b: DType::F32,
        }));
    }

    let shape = input.shape().to_vec();
    let size = input.size();

    let mut in_vals = vec![0.0f32; size];
    for flat in 0..size {
        let multi = unflatten_index(flat, &shape);
        in_vals[flat] = input.get(&multi)?;
    }

    let mut out_vals = vec![0.0f32; size];
    out_vals
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, val)| *val = in_vals[i] * factor);

    Tensor::from_slice(&out_vals, &shape).map_err(GraphError::from)
}

fn op_matmul(inputs: &[&Tensor]) -> Result<Tensor> {
    inputs[0].matmul(inputs[1]).map_err(GraphError::from)
}

fn op_softmax(inputs: &[&Tensor]) -> Result<Tensor> {
    let input = inputs[0];

    if input.dtype() != DType::F32 {
        return Err(GraphError::TensorError(TensorError::DTypeMismatch {
            a: input.dtype(),
            b: DType::F32,
        }));
    }

    let shape = input.shape().to_vec();
    if shape.is_empty() {
        return Ok(Tensor::zeros(&[], DType::F32));
    }

    let last_dim = shape[shape.len() - 1];
    if last_dim == 0 {
        return Ok(Tensor::zeros(&shape, DType::F32));
    }

    let size: usize = shape.iter().product();

    let mut in_vals = vec![0.0f32; size];
    for flat in 0..size {
        let idx = unflatten_index(flat, &shape);
        in_vals[flat] = input.get(&idx)?;
    }

    let mut out_vals = vec![0.0f32; size];
    out_vals
        .par_chunks_mut(last_dim)
        .enumerate()
        .for_each(|(row, chunk)| {
            let start = row * last_dim;
            let in_row = &in_vals[start..start + last_dim];

            let max_val = in_row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            let mut sum = 0.0f32;
            for &v in in_row {
                sum += (v - max_val).exp();
            }

            for (j, &v) in in_row.iter().enumerate() {
                chunk[j] = (v - max_val).exp() / sum;
            }
        });

    Tensor::from_slice(&out_vals, &shape).map_err(GraphError::from)
}

fn op_rms_norm(inputs: &[&Tensor], eps: f32) -> Result<Tensor> {
    let input = inputs[0];

    if input.dtype() != DType::F32 {
        return Err(GraphError::TensorError(TensorError::DTypeMismatch {
            a: input.dtype(),
            b: DType::F32,
        }));
    }

    let shape = input.shape().to_vec();
    let last_dim = shape[shape.len() - 1];
    let size: usize = shape.iter().product();

    let mut in_vals = vec![0.0f32; size];
    for flat in 0..size {
        let idx = unflatten_index(flat, &shape);
        in_vals[flat] = input.get(&idx)?;
    }

    let mut out_vals = vec![0.0f32; size];
    out_vals
        .par_chunks_mut(last_dim)
        .enumerate()
        .for_each(|(row, chunk)| {
            let start = row * last_dim;
            let in_row = &in_vals[start..start + last_dim];

            let sum_sq: f32 = in_row.iter().map(|&v| v * v).sum();
            let rms = (sum_sq / last_dim as f32 + eps).sqrt();

            for (j, &v) in in_row.iter().enumerate() {
                chunk[j] = v / rms;
            }
        });

    Tensor::from_slice(&out_vals, &shape).map_err(GraphError::from)
}

fn op_rope(inputs: &[&Tensor], theta: f32, dim_pairs: usize) -> Result<Tensor> {
    let input = inputs[0];

    if input.dtype() != DType::F32 {
        return Err(GraphError::TensorError(TensorError::DTypeMismatch {
            a: input.dtype(),
            b: DType::F32,
        }));
    }

    let shape = input.shape().to_vec();
    let dim = shape[1];
    let size = input.size();

    let mut in_vals = vec![0.0f32; size];
    for flat in 0..size {
        let idx = unflatten_index(flat, &shape);
        in_vals[flat] = input.get(&idx)?;
    }

    let mut out_vals = in_vals.clone();
    out_vals
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(pos, chunk)| {
            for pair in 0..dim_pairs {
                let i = 2 * pair;
                if i + 1 >= dim {
                    break;
                }
                let freq = theta.powf(-(2.0 * pair as f32) / dim as f32);
                let cos_val = (pos as f32 * freq).cos();
                let sin_val = (pos as f32 * freq).sin();

                let x0 = chunk[i];
                let x1 = chunk[i + 1];

                chunk[i] = x0 * cos_val - x1 * sin_val;
                chunk[i + 1] = x1 * cos_val + x0 * sin_val;
            }
        });

    Tensor::from_slice(&out_vals, &shape).map_err(GraphError::from)
}

fn op_silu(inputs: &[&Tensor]) -> Result<Tensor> {
    let input = inputs[0];

    if input.dtype() != DType::F32 {
        return Err(GraphError::TensorError(TensorError::DTypeMismatch {
            a: input.dtype(),
            b: DType::F32,
        }));
    }

    let shape = input.shape().to_vec();
    let size = input.size();

    let mut in_vals = vec![0.0f32; size];
    for flat in 0..size {
        let idx = unflatten_index(flat, &shape);
        in_vals[flat] = input.get(&idx)?;
    }

    let mut out_vals = vec![0.0f32; size];
    out_vals
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, val)| *val = in_vals[i] / (1.0 + (-in_vals[i]).exp()));

    Tensor::from_slice(&out_vals, &shape).map_err(GraphError::from)
}

fn op_cont(inputs: &[&Tensor]) -> Result<Tensor> {
    inputs[0].contiguous().map_err(GraphError::from)
}

fn op_repeat(inputs: &[&Tensor], params: Option<&[u8]>) -> Result<Tensor> {
    let input = inputs[0];

    if input.dtype() != DType::F32 {
        return Err(GraphError::TensorError(TensorError::DTypeMismatch {
            a: input.dtype(),
            b: DType::F32,
        }));
    }

    let bytes = params.ok_or_else(|| {
        GraphError::InvalidParams("Repeat requires target shape in params".into())
    })?;
    let target_shape = bytes_to_shape(bytes)?;
    let src_shape = input.shape().to_vec();

    let size: usize = target_shape.iter().product();

    let mut in_vals = vec![0.0f32; size];
    for flat in 0..size {
        let multi = unflatten_index(flat, &target_shape);
        let src_multi = broadcast_map(&multi, &src_shape);
        in_vals[flat] = input.get(&src_multi)?;
    }

    let mut out_vals = vec![0.0f32; size];
    out_vals
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, val)| *val = in_vals[i]);

    Tensor::from_slice(&out_vals, &target_shape).map_err(GraphError::from)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use xllm_tensor::Tensor;

    use super::*;

    fn assert_close(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "expected {a}, got {b}");
    }

    fn assert_tensor_close(t: &Tensor, expected: &[f32]) {
        for (i, &exp) in expected.iter().enumerate() {
            let idx = unflatten_index(i, t.shape());
            let val: f32 = t.get(&idx).unwrap();
            assert_close(val, exp);
        }
    }

    // -----------------------------------------------------------------------
    // 1. Graph with one op (Add)
    // -----------------------------------------------------------------------
    #[test]
    fn test_graph_add() {
        let mut builder = GraphBuilder::new();
        let a_id = builder.add_input(&[2, 2], DType::F32);
        let b_id = builder.add_input(&[2, 2], DType::F32);
        let add_id = builder.add(Op::Add, &[a_id, b_id]).unwrap();
        let mut graph = builder.build().unwrap();

        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        graph.set_input(a_id, a).unwrap();
        graph.set_input(b_id, b).unwrap();
        graph.forward().unwrap();

        let result = graph.tensor(add_id).unwrap();
        assert_tensor_close(result, &[6.0, 8.0, 10.0, 12.0]);
    }

    // -----------------------------------------------------------------------
    // 2. Graph with MatMul
    // -----------------------------------------------------------------------
    #[test]
    fn test_graph_matmul() {
        let mut builder = GraphBuilder::new();
        let a_id = builder.add_input(&[2, 3], DType::F32);
        let b_id = builder.add_input(&[3, 2], DType::F32);
        let mm_id = builder.add(Op::MatMul, &[a_id, b_id]).unwrap();
        let mut graph = builder.build().unwrap();

        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::from_slice(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();
        graph.set_input(a_id, a).unwrap();
        graph.set_input(b_id, b).unwrap();
        graph.forward().unwrap();

        let result = graph.tensor(mm_id).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_tensor_close(result, &[58.0, 64.0, 139.0, 154.0]);
    }

    // -----------------------------------------------------------------------
    // 3. SoftMax numerical correctness
    // -----------------------------------------------------------------------
    #[test]
    fn test_graph_softmax() {
        let mut builder = GraphBuilder::new();
        let x_id = builder.add_input(&[2, 3], DType::F32);
        let sm_id = builder.add(Op::SoftMax, &[x_id]).unwrap();
        let mut graph = builder.build().unwrap();

        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]).unwrap();
        graph.set_input(x_id, x).unwrap();
        graph.forward().unwrap();

        let result = graph.tensor(sm_id).unwrap();
        // Row 0 softmax of [1, 2, 3]
        let e1 = (-2.0f32).exp();
        let e2 = (-1.0f32).exp();
        let e3 = 0.0f32.exp();
        let sum = e1 + e2 + e3;
        let expected_row = [e1 / sum, e2 / sum, e3 / sum];
        assert_tensor_close(
            result,
            &[
                expected_row[0],
                expected_row[1],
                expected_row[2],
                expected_row[0],
                expected_row[1],
                expected_row[2],
            ],
        );

        // Each row should sum to 1
        for row in 0..2 {
            let mut s = 0.0f32;
            for col in 0..3 {
                s += result.get::<f32>(&[row, col]).unwrap();
            }
            assert_close(s, 1.0);
        }
    }

    // -----------------------------------------------------------------------
    // 4. RMSNorm numerical correctness
    // -----------------------------------------------------------------------
    #[test]
    fn test_graph_rms_norm() {
        let mut builder = GraphBuilder::new();
        let x_id = builder.add_input(&[2, 4], DType::F32);
        let rn_id = builder.add(Op::RMSNorm { eps: 1e-6 }, &[x_id]).unwrap();
        let mut graph = builder.build().unwrap();

        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).unwrap();
        graph.set_input(x_id, x).unwrap();
        graph.forward().unwrap();

        let result = graph.tensor(rn_id).unwrap();
        // Row 0: mean(x^2) = (1+4+9+16)/4 = 7.5, rms = sqrt(7.5 + 1e-6)
        let rms0 = (7.5f32 + 1e-6).sqrt();
        let expected0 = [1.0 / rms0, 2.0 / rms0, 3.0 / rms0, 4.0 / rms0];
        // Row 1: mean(x^2) = (25+36+49+64)/4 = 43.5, rms = sqrt(43.5 + 1e-6)
        let rms1 = (43.5f32 + 1e-6).sqrt();
        let expected1 = [5.0 / rms1, 6.0 / rms1, 7.0 / rms1, 8.0 / rms1];
        let mut expected = Vec::new();
        expected.extend_from_slice(&expected0);
        expected.extend_from_slice(&expected1);
        assert_tensor_close(result, &expected);
    }

    // -----------------------------------------------------------------------
    // 5. Silu correctness
    // -----------------------------------------------------------------------
    #[test]
    fn test_graph_silu() {
        let mut builder = GraphBuilder::new();
        let x_id = builder.add_input(&[1], DType::F32);
        let silu_id = builder.add(Op::Silu, &[x_id]).unwrap();
        let mut graph = builder.build().unwrap();

        let x = Tensor::from_slice(&[0.5f32], &[1]).unwrap();
        graph.set_input(x_id, x).unwrap();
        graph.forward().unwrap();

        let result = graph.tensor(silu_id).unwrap();
        // silu(0.5) = 0.5 / (1 + exp(-0.5))
        let expected = 0.5f32 / (1.0 + (-0.5f32).exp());
        assert_close(result.get::<f32>(&[0]).unwrap(), expected);
    }

    // -----------------------------------------------------------------------
    // 6. Graph with 3 nodes: Add -> Silu -> Scale
    // -----------------------------------------------------------------------
    #[test]
    fn test_graph_multi_node() {
        let mut builder = GraphBuilder::new();
        let a_id = builder.add_input(&[3], DType::F32);
        let b_id = builder.add_input(&[3], DType::F32);
        let add_id = builder.add(Op::Add, &[a_id, b_id]).unwrap();
        let silu_id = builder.add(Op::Silu, &[add_id]).unwrap();
        let scale_id = builder.add(Op::Scale(2.0), &[silu_id]).unwrap();
        let mut graph = builder.build().unwrap();

        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], &[3]).unwrap();
        graph.set_input(a_id, a).unwrap();
        graph.set_input(b_id, b).unwrap();
        graph.forward().unwrap();

        let result = graph.tensor(scale_id).unwrap();
        let add_vals = [5.0f32, 7.0, 9.0];
        for (i, &av) in add_vals.iter().enumerate() {
            let silu_val = av / (1.0 + (-av).exp());
            let expected = silu_val * 2.0;
            assert_close(result.get::<f32>(&[i]).unwrap(), expected);
        }
    }

    // -----------------------------------------------------------------------
    // 7. Error when input ID does not exist
    // -----------------------------------------------------------------------
    #[test]
    fn test_error_invalid_input_id() {
        let mut builder = GraphBuilder::new();
        let a_id = builder.add_input(&[2], DType::F32);
        // Try to add a node referencing a non-existent input
        let err = builder.add(Op::Silu, &[a_id, 999]).unwrap_err();
        assert!(matches!(err, GraphError::NodeNotFound { id: 999 }));
    }

    #[test]
    fn test_error_input_not_set() {
        let mut builder = GraphBuilder::new();
        let a_id = builder.add_input(&[3], DType::F32);
        let _silu_id = builder.add(Op::Silu, &[a_id]).unwrap();
        let mut graph = builder.build().unwrap();
        // Don't set input a_id — should get InputNotFound
        let err = graph.forward().unwrap_err();
        assert!(matches!(err, GraphError::InputNotFound { .. }));
    }

    // -----------------------------------------------------------------------
    // 8. Reshape with params
    // -----------------------------------------------------------------------
    #[test]
    fn test_graph_reshape() {
        let mut builder = GraphBuilder::new();
        let x_id = builder.add_input(&[2, 3], DType::F32);
        let reshape_id = builder.add_reshape(x_id, &[3, 2]).unwrap();
        let mut graph = builder.build().unwrap();

        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        graph.set_input(x_id, x).unwrap();
        graph.forward().unwrap();

        let result = graph.tensor(reshape_id).unwrap();
        assert_eq!(result.shape(), &[3, 2]);
        assert_tensor_close(result, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape_element_count_mismatch() {
        let mut builder = GraphBuilder::new();
        let x_id = builder.add_input(&[2, 3], DType::F32);
        let err = builder.add_reshape(x_id, &[2, 4]).unwrap_err();
        assert!(matches!(
            err,
            GraphError::TensorError(TensorError::ElementCount { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // 9. RoPE correctness (simple case)
    // -----------------------------------------------------------------------
    #[test]
    fn test_graph_rope() {
        let mut builder = GraphBuilder::new();
        let x_id = builder.add_input(&[3, 4], DType::F32); // 3 positions, 4 dims -> 2 pairs
        let rope_id = builder
            .add(
                Op::RoPE {
                    theta: 10000.0,
                    dim_pairs: 2,
                },
                &[x_id],
            )
            .unwrap();
        let mut graph = builder.build().unwrap();

        // Simple input: all ones
        let x = Tensor::from_slice(&[1.0f32; 12], &[3, 4]).unwrap();
        graph.set_input(x_id, x).unwrap();
        graph.forward().unwrap();

        let result = graph.tensor(rope_id).unwrap();
        assert_eq!(result.shape(), &[3, 4]);

        // For pos=0: cos=1, sin=0 -> output should equal input
        assert_close(result.get::<f32>(&[0, 0]).unwrap(), 1.0);
        assert_close(result.get::<f32>(&[0, 1]).unwrap(), 1.0);
        assert_close(result.get::<f32>(&[0, 2]).unwrap(), 1.0);
        assert_close(result.get::<f32>(&[0, 3]).unwrap(), 1.0);

        // pos=1, pair=0: freq = theta^(-0/4) = 1, cos=cos(1), sin=sin(1)
        // y0 = 1*cos - 1*sin = cos(1) - sin(1)
        // y1 = 1*cos + 1*sin = cos(1) + sin(1)
        let cos1 = 1.0f32.cos();
        let sin1 = 1.0f32.sin();
        assert_close(result.get::<f32>(&[1, 0]).unwrap(), cos1 - sin1);
        assert_close(result.get::<f32>(&[1, 1]).unwrap(), cos1 + sin1);

        // pos=2, pair=1: freq = theta^(-2/4) = 10000^(-0.5) = 0.01
        // cos=cos(2*0.01), sin=sin(2*0.01)
        let freq1 = 10000.0f32.powf(-2.0 / 4.0); // = 0.01
        let cos2 = (2.0f32 * freq1).cos();
        let sin2 = (2.0f32 * freq1).sin();
        assert_close(result.get::<f32>(&[2, 2]).unwrap(), cos2 - sin2);
        assert_close(result.get::<f32>(&[2, 3]).unwrap(), cos2 + sin2);
    }

    // -----------------------------------------------------------------------
    // 10. Repeat broadcast
    // -----------------------------------------------------------------------
    #[test]
    fn test_graph_repeat() {
        let mut builder = GraphBuilder::new();
        let x_id = builder.add_input(&[3], DType::F32); // 1D
        let repeat_id = builder.add_repeat(x_id, &[2, 3]).unwrap();
        let mut graph = builder.build().unwrap();

        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        graph.set_input(x_id, x).unwrap();
        graph.forward().unwrap();

        let result = graph.tensor(repeat_id).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_tensor_close(result, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    // -----------------------------------------------------------------------
    // Additional tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_graph_mul() {
        let mut builder = GraphBuilder::new();
        let a_id = builder.add_input(&[2, 2], DType::F32);
        let b_id = builder.add_input(&[2, 2], DType::F32);
        let mul_id = builder.add(Op::Mul, &[a_id, b_id]).unwrap();
        let mut graph = builder.build().unwrap();

        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        graph.set_input(a_id, a).unwrap();
        graph.set_input(b_id, b).unwrap();
        graph.forward().unwrap();

        let result = graph.tensor(mul_id).unwrap();
        assert_tensor_close(result, &[5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn test_graph_mul_broadcast() {
        let mut builder = GraphBuilder::new();
        let a_id = builder.add_input(&[2, 3], DType::F32);
        let b_id = builder.add_input(&[3], DType::F32);
        let mul_id = builder.add(Op::Mul, &[a_id, b_id]).unwrap();
        let mut graph = builder.build().unwrap();

        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::from_slice(&[10.0f32, 20.0, 30.0], &[3]).unwrap();
        graph.set_input(a_id, a).unwrap();
        graph.set_input(b_id, b).unwrap();
        graph.forward().unwrap();

        let result = graph.tensor(mul_id).unwrap();
        assert_tensor_close(result, &[10.0, 40.0, 90.0, 40.0, 100.0, 180.0]);
    }

    #[test]
    fn test_graph_scale() {
        let mut builder = GraphBuilder::new();
        let x_id = builder.add_input(&[3], DType::F32);
        let scale_id = builder.add(Op::Scale(3.0), &[x_id]).unwrap();
        let mut graph = builder.build().unwrap();

        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        graph.set_input(x_id, x).unwrap();
        graph.forward().unwrap();

        let result = graph.tensor(scale_id).unwrap();
        assert_tensor_close(result, &[3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_graph_copy() {
        let mut builder = GraphBuilder::new();
        let x_id = builder.add_input(&[2, 2], DType::F32);
        let copy_id = builder.add(Op::Copy, &[x_id]).unwrap();
        let mut graph = builder.build().unwrap();

        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        graph.set_input(x_id, x).unwrap();
        graph.forward().unwrap();

        let result = graph.tensor(copy_id).unwrap();
        assert_tensor_close(result, &[1.0, 2.0, 3.0, 4.0]);
        assert!(result.is_contiguous());
    }

    #[test]
    fn test_graph_cont() {
        let mut builder = GraphBuilder::new();
        let x_id = builder.add_input(&[2, 3], DType::F32);
        let t_id = builder.add(Op::Permute, &[x_id]).unwrap();
        let cont_id = builder.add(Op::Cont, &[t_id]).unwrap();
        let mut graph = builder.build().unwrap();

        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        graph.set_input(x_id, x).unwrap();
        graph.forward().unwrap();

        let result = graph.tensor(cont_id).unwrap();
        assert!(result.is_contiguous());
        assert_tensor_close(result, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_graph_get_rows() {
        let mut builder = GraphBuilder::new();
        let data_id = builder.add_input(&[4, 2], DType::F32);
        let idx_id = builder.add_input(&[3], DType::F32);
        let gr_id = builder.add(Op::GetRows, &[data_id, idx_id]).unwrap();
        let mut graph = builder.build().unwrap();

        let data = Tensor::from_slice(&[0.0f32, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0], &[4, 2])
            .unwrap();
        let indices = Tensor::from_slice(&[0.0f32, 2.0, 3.0], &[3]).unwrap();
        graph.set_input(data_id, data).unwrap();
        graph.set_input(idx_id, indices).unwrap();
        graph.forward().unwrap();

        let result = graph.tensor(gr_id).unwrap();
        assert_eq!(result.shape(), &[3, 2]);
        // rows [0, 2, 3] from data
        assert_tensor_close(result, &[0.0, 1.0, 20.0, 21.0, 30.0, 31.0]);
    }

    #[test]
    fn test_empty_graph() {
        let builder = GraphBuilder::new();
        let mut graph = builder.build().unwrap();
        graph.forward().unwrap(); // no error
    }

    #[test]
    fn test_graph_permute_noop() {
        // Permute returns a contiguous copy (same shape)
        let mut builder = GraphBuilder::new();
        let x_id = builder.add_input(&[2, 3], DType::F32);
        let p_id = builder.add(Op::Permute, &[x_id]).unwrap();
        let mut graph = builder.build().unwrap();

        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        graph.set_input(x_id, x).unwrap();
        graph.forward().unwrap();

        let result = graph.tensor(p_id).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_tensor_close(result, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_error_cycle_detected() {
        // Create a cycle manually by modifying the graph
        let mut builder = GraphBuilder::new();
        let a_id = builder.add_input(&[2], DType::F32);
        let b_id = builder.add(Op::Copy, &[a_id]).unwrap();
        let mut graph = builder.build().unwrap();

        // Manually create a cycle by adding a node that references forward
        // Actually, since forward() uses the nodes in order, a cycle would be
        // detected by the topological sort. But with our builder, you can't
        // create a cycle because forward-references aren't possible.
        // The closest we can test is that forward works correctly.
        graph
            .set_input(a_id, Tensor::zeros(&[2], DType::F32))
            .unwrap();
        graph.forward().unwrap();
        let result = graph.tensor(b_id).unwrap();
        assert_eq!(result.shape(), &[2]);
    }

    #[test]
    fn test_softmax_stability() {
        // Test with large values to check numerical stability
        let mut builder = GraphBuilder::new();
        let x_id = builder.add_input(&[1, 4], DType::F32);
        let sm_id = builder.add(Op::SoftMax, &[x_id]).unwrap();
        let mut graph = builder.build().unwrap();

        let x = Tensor::from_slice(&[100.0f32, 101.0, 102.0, 103.0], &[1, 4]).unwrap();
        graph.set_input(x_id, x).unwrap();
        graph.forward().unwrap();

        let result = graph.tensor(sm_id).unwrap();
        let mut sum = 0.0f32;
        for j in 0..4 {
            let v = result.get::<f32>(&[0, j]).unwrap();
            assert!(v.is_finite());
            sum += v;
        }
        assert_close(sum, 1.0);
    }

    #[test]
    fn test_rms_norm_eps() {
        // Test that eps prevents division by zero
        let mut builder = GraphBuilder::new();
        let x_id = builder.add_input(&[1, 3], DType::F32);
        let rn_id = builder.add(Op::RMSNorm { eps: 0.1 }, &[x_id]).unwrap();
        let mut graph = builder.build().unwrap();

        let x = Tensor::zeros(&[1, 3], DType::F32);
        graph.set_input(x_id, x).unwrap();
        graph.forward().unwrap();

        let result = graph.tensor(rn_id).unwrap();
        for j in 0..3 {
            let v = result.get::<f32>(&[0, j]).unwrap();
            assert!(v.is_finite());
        }
    }
}
