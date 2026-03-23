import torch
import time
import triton
import triton.language as tl

@triton.jit
def gemm_kernel_coalesced_smem(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    row = pid // N
    col = pid % N

    acc = 0.0

    for k0 in range(0, K, BLOCK_K):
        # Create range [0, BLOCK_K)
        offsets = tl.arange(0, BLOCK_K)
        k = k0 + offsets

        # Mask for boundary
        mask = k < K

        # ---- "Shared memory" loads (block load) ----
        a_smem = tl.load(
            A_ptr + row * stride_am + k * stride_ak,
            mask=mask,
            other=0.0
        )
        b_smem = tl.load(
            B_ptr + k * stride_bk + col * stride_bn,
            mask=mask,
            other=0.0
        )

        # ---- Compute on cached block ----
        acc += tl.sum(a_smem * b_smem)

    tl.store(C_ptr + row * stride_cm + col * stride_cn, acc)

def run_coalesced_smem_gemm(A, B, DEVICE):
    """
    Runs 1D K-blocking coalesce + shared memory blocked GEMM in Triton with warmup,
    timing, and GFLOPS calculation.
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

    # ---- 1D grid (unchanged) ----
    grid = (M * N,)

    BLOCK_K = 32  # required for shared memory blocking

    # ---- Warmup ----
    gemm_kernel_coalesced_smem[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_K,
    )
    torch.cuda.synchronize()

    # ---- Timing ----
    start = time.time()
    gemm_kernel_coalesced_smem[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_K,
    )
    torch.cuda.synchronize()
    end = time.time()

    runtime = end - start
    gflops = 2 * M * N * K / runtime / 1e9

    print(f"Coalesced + SMEM GEMM runtime: {runtime*1000:.3f} ms")
    print(f"Performance: {gflops:.2f} GFLOPS/s")

    return C, runtime, gflops