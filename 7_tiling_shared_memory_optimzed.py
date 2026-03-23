import torch
import time
import triton
import triton.language as tl

@triton.jit
def gemm_kernel_2d_tiling_smem_optimized(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,  # rows per tile
    BLOCK_N: tl.constexpr,  # columns per tile
    BLOCK_K: tl.constexpr,  # K-blocking
):
    # ---- 2D program IDs ----
    pid_m = tl.program_id(0)  # tile row
    pid_n = tl.program_id(1)  # tile column

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    # ---- Row and column ranges for this tile ----
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    # ---- Accumulator ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- K loop with blocking ----
    for k0 in range(0, K, BLOCK_K):
        # K-block indices
        k_range = tl.arange(0, BLOCK_K)
        k = k0 + k_range
        k_mask = k < K

        # ---- Load A tile once into registers ----
        a_ptrs = A_ptr + rows[:, None] * stride_am + k[None, :] * stride_ak
        # vectorized load along K
        a_tile = tl.load(a_ptrs, mask=(rows[:, None] < M) & k_mask[None, :], other=0.0)

        # ---- Load B tile once into registers ----
        b_ptrs = B_ptr + k[:, None] * stride_bk + cols[None, :] * stride_bn
        # vectorized load along K
        b_tile = tl.load(b_ptrs, mask=k_mask[:, None] & (cols[None, :] < N), other=0.0)

        # ---- Compute using loaded tiles ----
        # Here each element of a_tile and b_tile is reused across BLOCK_M x BLOCK_N
        acc += tl.dot(a_tile, b_tile)

    # ---- Store C tile ----
    c_ptrs = C_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)

def run_2d_tiling_smem_optimized_gemm_optimized(A, B, DEVICE, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
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
    gemm_kernel_2d_tiling_smem_optimized[grid](
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
    gemm_kernel_2d_tiling_smem_optimized[grid](
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

    print(f"2D Tiling + SMEM GEMM runtime: {runtime*1000:.3f} ms")
    print(f"Performance: {gflops:.2f} GFLOPS/s")

    return C, runtime, gflops