# xllm-bitnet

1-bit ternary kernels for BitNet b1.58 inference.

Implements ternary weight format {-1, 0, +1} with pure integer
addition/subtraction — no floating-point multiplications. Uses
lookup-table-based matmul kernels following the T-MAC approach.

## License

MIT OR Apache-2.0
