## 🧠 Kernels Summary

| Kernel | Description | Strategy | Performance Impact |
|--------|------------|----------|--------------------|
| `1 gemm_kernel` | Baseline implementation | Scalar compute | ❌ Memory-bound, no reuse |
| `2 gemm_kernel_coalesced` | Linearized thread mapping | Memory coalescing | ⚠️ Slight improvement |
| `3 gemm_kernel_coalesced_smem` | K-blocked scalar | 1D-blocking | ✅ Better locality |
| `4 gemm_kernel_shared` | Tiled GEMM | 2D block loads` | ✅ Major speedup |
| `5 gemm_kernel_1d_tiling_coalesced_smem` | Column-tiled GEMM | 1D tiling | ✅ Efficient memory access |
| `6 gemm_kernel_2d_tiling_smem` | Fully tiled GEMM | 2D tiling | 🚀 Better performance |
| `7 gemm_kernel_2d_tiling_smem_optimized` | Optimized tiled GEMM | Vectorization + reuse | 🚀🚀 High performance |
