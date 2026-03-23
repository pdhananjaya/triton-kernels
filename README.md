## 🧠 Kernels Summary

| Kernel | Description | Strategy | Performance Impact |
|--------|------------|----------|--------------------|
| `gemm_kernel` | Baseline implementation | Scalar compute | ❌ Memory-bound, no reuse |
| `gemm_kernel_coalesced` | Linearized thread mapping | Memory coalescing | ⚠️ Slight improvement |
| `gemm_kernel_coalesced_smem` | K-blocked scalar | 1D-blocking | ✅ Better locality |
| `gemm_kernel_shared` | Tiled GEMM | 2D block loads` | ✅ Major speedup |
| `gemm_kernel_1d_tiling_coalesced_smem` | Column-tiled GEMM | 1D tiling | ✅ Efficient memory access |
| `gemm_kernel_2d_tiling_smem` | Fully tiled GEMM | 2D tiling | 🚀 High performance |
| `gemm_kernel_2d_tiling_smem_optimized` | Optimized tiled GEMM | Vectorization + reuse | 🚀🚀 Near-optimal |
