import torch
import time
import triton
import triton.language as tl

@triton.jit
def gemm_kernel_coalesced(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
):
    pid = tl.program_id(0)

    # Flattened program id mapped to (row, col)
    row = pid // N
    col = pid % N

    acc = 0.0
    for k in range(K):
        a = tl.load(A_ptr + row * stride_am + k * stride_ak)
        b = tl.load(B_ptr + k * stride_bk + col * stride_bn)
        acc += a * b

    tl.store(C_ptr + row * stride_cm + col * stride_cn, acc)

def run_coalesced_gemm(A, B, DEVICE):
    """
    Runs Global memory coalescing GEMM in Triton with warmup, timing, and GFLOPS calculation.
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

    # ---- 1D grid for coalesced access ----
    grid = (M * N,)

    # ---- Warmup ----
    gemm_kernel_coalesced[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    torch.cuda.synchronize()

    # ---- Timing ----
    start = time.time()
    gemm_kernel_coalesced[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    torch.cuda.synchronize()
    end = time.time()

    runtime = end - start
    gflops = 2 * M * N * K / runtime / 1e9

    print(f"Naive GEMM runtime: {runtime*1000:.3f} ms")
    print(f"Performance: {gflops:.2f} GFLOPS/s")

    return C, runtime, gflops