import torch
import time
import triton
import triton.language as tl


@triton.jit
def gemm_kernel_1d_tiling_coalesced_smem(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    # ---- 1D tiling over N (columns) ----
    num_pid_n = N // BLOCK_N
    row = pid // num_pid_n
    col_start = (pid % num_pid_n) * BLOCK_N

    cols = col_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # ---- K blocking (shared memory style) ----
    for k0 in range(0, K, BLOCK_K):
        offsets = tl.arange(0, BLOCK_K)
        k = k0 + offsets
        k_mask = k < K

        # Load A block (broadcast over columns)
        a = tl.load(
            A_ptr + row * stride_am + k * stride_ak,
            mask=k_mask,
            other=0.0
        )  # shape: (BLOCK_K,)

        # Load B block (coalesced across columns)
        b = tl.load(
            B_ptr + k[:, None] * stride_bk + cols[None, :] * stride_bn,
            mask=k_mask[:, None],
            other=0.0
        )  # shape: (BLOCK_K, BLOCK_N)

        # ---- Compute (same idea as your smem kernel) ----
        acc += tl.sum(a[:, None] * b, axis=0)

    # ---- Store ----
    tl.store(C_ptr + row * stride_cm + cols * stride_cn, acc)

def run_1d_tiling_coalesced_smem_gemm(A, B, DEVICE):
    """
    Runs 1D tiling + coalesced + shared-memory blocked GEMM in Triton
    with warmup, timing, and GFLOPS calculation.

    Returns:
        C: output matrix
        runtime: execution time in seconds
        gflops: achieved GFLOPS/s
    """
    import torch, time

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    C = torch.zeros((M, N), device=DEVICE, dtype=torch.float32)

    # ---- Tiling parameters ----
    BLOCK_N = 32
    BLOCK_K = 32

    # ---- 1D grid ----
    grid = (M * (N // BLOCK_N),)

    # ---- Warmup ----
    gemm_kernel_1d_tiling_coalesced_smem[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_N,
        BLOCK_K,
    )
    torch.cuda.synchronize()

    # ---- Timing ----
    start = time.time()
    gemm_kernel_1d_tiling_coalesced_smem[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_N,
        BLOCK_K,
    )
    torch.cuda.synchronize()
    end = time.time()

    runtime = end - start
    gflops = 2 * M * N * K / runtime / 1e9

    print(f"1D Tiling + Coalesced + SMEM GEMM runtime: {runtime*1000:.3f} ms")
    print(f"Performance: {gflops:.2f} GFLOPS/s")

    return C, runtime, gflops