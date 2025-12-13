#include <torch/extension.h>
#include "matadd.h"

void torch_launch_matadd(torch::Tensor &c,
                         const torch::Tensor &a,
                         const torch::Tensor &b,
                         int64_t n) {
    // 检查张量是否在 CUDA 上
    TORCH_CHECK(c.is_cuda(), "Tensor c must be on CUDA");
    TORCH_CHECK(a.is_cuda(), "Tensor a must be on CUDA");
    TORCH_CHECK(b.is_cuda(), "Tensor b must be on CUDA");
    
    // 检查张量形状
    TORCH_CHECK(c.sizes() == a.sizes() && a.sizes() == b.sizes(), 
                "All tensors must have the same shape");
    
    launch_matadd((float *)c.data_ptr(),
                  (const float *)a.data_ptr(),
                  (const float *)b.data_ptr(),
                  static_cast<int>(n));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_matadd",
          &torch_launch_matadd,
          "Matrix add kernel wrapper");
} 