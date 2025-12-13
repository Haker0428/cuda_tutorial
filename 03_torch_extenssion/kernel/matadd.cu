#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

__global__ void MatAdd(float* c,
                       const float* a,
                       const float* b,
                       int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = j * n + i;
    if (i < n && j < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_matadd(float* c,
                   const float* a,
                   const float* b,
                   int n) {
    dim3 block(16, 16);
    // 向上取整以确保覆盖所有元素
    dim3 grid((n + block.x - 1) / block.x, 
              (n + block.y - 1) / block.y);

    MatAdd<<<grid, block>>>(c, a, b, n);
    
    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel execution error: %s\n", cudaGetErrorString(err));
    }
}

