import torch
import time
import triton
import triton.language as tl

@triton.jit
def gemm_kernel_shared(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_range = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # mask for boundary handling
    row_mask = row < M
    col_mask = col < N

    for k in range(0, K, BLOCK_K):
        k_mask = k + k_range < K

        # ---- Load tiles (cached in SRAM) ----
        a_ptrs = A_ptr + row[:, None] * stride_am + (k + k_range)[None, :] * stride_ak
        b_ptrs = B_ptr + (k + k_range)[:, None] * stride_bk + col[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & col_mask[None, :], other=0.0)

        # ---- Compute on tiles ----
        acc += tl.dot(a, b)

    # ---- Store result ----
    c_ptrs = C_ptr + row[:, None] * stride_cm + col[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=row_mask[:, None] & col_mask[None, :])

import torch
import time

def run_shared_gemm(A, B, DEVICE, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    """
    Runs 2D tiling + shared-memory style GEMM in Triton.

    Args:
        A, B: input matrices (torch tensors)
        DEVICE: torch device (e.g., "cuda")
        BLOCK_M, BLOCK_N, BLOCK_K: tiling sizes

    Returns:
        C: output matrix
        runtime: execution time in seconds
        gflops: achieved GFLOPS/s
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    # Allocate output
    C = torch.zeros((M, N), device=DEVICE, dtype=torch.float32)

    # 2D grid (number of tiles)
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    # ---- Warmup ----
    gemm_kernel_shared[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    torch.cuda.synchronize()

    # ---- Timing ----
    start = time.time()
    gemm_kernel_shared[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    torch.cuda.synchronize()
    end = time.time()

    runtime = end - start
    gflops = 2 * M * N * K / runtime / 1e9

    print(f"Shared-memory GEMM runtime: {runtime*1000:.3f} ms")
    print(f"Performance: {gflops:.2f} GFLOPS/s")

    return C, runtime, gflops
