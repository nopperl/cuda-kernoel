# Implements a Triton kernel for head-wise RMS normalization.
# This is used for normalizing the Q and K projections prior to self attention.
# The input to the RMSNorm has a shape of (..., num_heads, head_size).
# The weights have a shape of (num_heads, head_size).
# The necessitates a reduction along the `head_size`.
# The Triton kernel is implemented such that one block operates on one head.

import torch

import triton
import triton.language as tl

def _rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor,
              eps: float) -> torch.Tensor:
    input_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(input_shape[:-1] + weight.shape)
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = hidden_states.to(input_dtype)
    hidden_states = weight * hidden_states
    return hidden_states.reshape(input_shape)

@triton.jit
def _qk_rms_norm_kernel(
X_ptr, X_row_stride,  # (num_tokens * num_heads, head_size
weights_ptr,  # (num_heads, head_size), stride == num_cols
num_heads: tl.constexpr,
num_cols: tl.constexpr,  # head_size
eps: tl.constexpr,
BLOCK_SIZE: tl.constexpr,
):
    # TODO: handle dtypes more robustly
    row_idx = tl.program_id(0)  # Every program processes one head of one token
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_cols

    X_ptr += row_idx * X_row_stride
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype
    # compute rms scaling in fp32 for accuracy
    X_row = X_row.to(tl.float32)
    head_idx = row_idx % num_heads
    weights_ptr += head_idx * num_cols
    weights_row = tl.load(weights_ptr + col_offsets, mask=mask, other=1)

    row_var = tl.sum(X_row * X_row, axis = 0) / num_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    X_row = X_row * inv_var
    X_row = X_row.to(X_row_dtype)
    # TODO: allow offset (intercept): X_row = X_row * (offset + weights_row)
    X_row = X_row * weights_row

    tl.store(X_ptr + col_offsets, X_row, mask=mask)


def qk_rms_norm(
    x,  # (*, [batch_size, sequence_length], num_heads * head_size)
    weights, # (num_heads, head_size)
    eps=1e-6):
    # TODO: instead of getting num_heads, head_size via weights, set hidden_size via config
    input_shape = x.shape
    num_heads, head_size = weights.shape
    x = x.reshape(-1, head_size)
    num_rows, _ = x.shape
    BLOCK_SIZE = triton.next_power_of_2(head_size)
    _qk_rms_norm_kernel[(num_rows,)](
            x, x.stride(0),
            weights,
            num_heads,
            head_size,
            eps,
            BLOCK_SIZE,
            num_warps=4,
    )
    return x.reshape(input_shape)
    

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['head_size'],
        x_vals=[64, 128, 256, 512],  # Different possible values for `x_name`.
        x_log=False,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'eager', 'torch.compile'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch eager', 'torch.compile'],  # Label name for the lines.
#        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='ms',  # Label name for the y-axis.
        plot_name='rms-norm-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark_qk_rms_norm(head_size, provider):
    DEVICE = "cuda"
    num_tokens = 2048
    num_heads = 16
    x = torch.rand((num_tokens, num_heads * head_size), device=DEVICE, dtype=torch.bfloat16)
    weights = torch.rand((16, head_size), device=DEVICE, dtype=torch.bfloat16)
    eps = 1e-6
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'eager':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: _rms_norm(x, weights, eps=eps), quantiles=quantiles)
    if provider == 'torch.compile':
        _rms_norm_opt = torch.compile(_rms_norm)
        # warmup compile
        _rms_norm_opt(x, weights, eps=eps)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: _rms_norm_opt(x, weights, eps=eps), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: qk_rms_norm(x, weights, eps=eps), quantiles=quantiles)
    bytes_ = x.numel() * x.element_size() + weights.numel() * weights.element_size()
    return ms, max_ms, min_ms

def test_qk_rms_norm(num_tokens=2048, num_heads=16, head_size=512, eps=1e-6, dtype=torch.float16, device="cuda", random_seed=341):
    torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    x = torch.randn((num_tokens, num_heads * head_size), device=device, dtype=dtype)
    weights = torch.randn((16, head_size), device=device, dtype=dtype)
    x2 = x.clone()
    qk_rms_norm(x, weights, eps=eps)
    x2 = _rms_norm(x2, weights, eps=eps)
    triton.testing.assert_close(x, x2)

if __name__ == "__main__":
    test_qk_rms_norm()
    benchmark_qk_rms_norm.run(print_data=True, show_plots=True)

