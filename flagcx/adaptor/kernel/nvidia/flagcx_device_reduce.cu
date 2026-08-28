/* 设备侧 reduce（Sum）kernel：dst += src，供 uniRunner 异构 allreduce 使用。
 * 由 cuda_adaptor.cc 的 cudaAdaptorReduceSum 通过 flagcxLaunchReduceSum 调用。
 * flagcxDataType_t: flagcxFloat16=6, flagcxFloat32=7, flagcxBfloat16=9
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__global__ void flagcxReduceSumKernelF32(const float *a, const float *b, float *out, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}
__global__ void flagcxReduceSumKernelF16(const __half *a, const __half *b, __half *out, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) out[i] = __hadd(a[i], b[i]);
}
__global__ void flagcxReduceSumKernelBF16(const __nv_bfloat16 *a, const __nv_bfloat16 *b, __nv_bfloat16 *out, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) out[i] = __hadd(a[i], b[i]);
}

extern "C" void flagcxLaunchReduceSum(void *dst, const void *src, size_t count,
                                      int datatype, void *stream) {
  cudaStream_t s = (cudaStream_t)stream;
  int threads = 256;
  size_t blocks = (count + threads - 1) / threads;
  switch (datatype) {
    case 7: /* flagcxFloat32 */
      flagcxReduceSumKernelF32<<<blocks, threads, 0, s>>>(
          (const float *)dst, (const float *)src, (float *)dst, count);
      break;
    case 6: /* flagcxFloat16 */
      flagcxReduceSumKernelF16<<<blocks, threads, 0, s>>>(
          (const __half *)dst, (const __half *)src, (__half *)dst, count);
      break;
    case 9: /* flagcxBfloat16 */
      flagcxReduceSumKernelBF16<<<blocks, threads, 0, s>>>(
          (const __nv_bfloat16 *)dst, (const __nv_bfloat16 *)src, (__nv_bfloat16 *)dst, count);
      break;
    default:
      break;
  }
}
