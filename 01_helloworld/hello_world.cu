/*
 * hello_world.cu
*/

#include <stdio.h>

// __global__ 告诉编译器这个是个可以在设备上执行的核函数
__global__ void hello_world(void)
{
    unsigned smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    printf("GPU: Hello World threadIdx.x %d blockIdx.x %d sm_id %d\n", threadIdx.x ,blockIdx.x, smid);
}

int main()
{
    printf("CPU: Hello World\n");

    hello_world<<<1, 200>>>();
    cudaDeviceSynchronize();  // 等待 GPU 完成执行
    cudaDeviceReset();
    return 0;
}