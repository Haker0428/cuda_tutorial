#include <torch/extension.h>
#include "flash_attention.h"

void torch_launch_flash_attention(torch::Tensor &output,
                                  const torch::Tensor &q,
                                  const torch::Tensor &k,
                                  const torch::Tensor &v,
                                  int64_t batch_size,
                                  int64_t num_heads,
                                  int64_t seq_len,
                                  int64_t head_dim,
                                  float scale) {
    // TODO: 添加张量检查
    // TORCH_CHECK(output.is_cuda(), "output must be on CUDA");
    // TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
    // TORCH_CHECK(k.is_cuda(), "k must be on CUDA");
    // TORCH_CHECK(v.is_cuda(), "v must be on CUDA");
    // TORCH_CHECK(output.dim() == 4, "output must be 4D tensor");
    // TORCH_CHECK(q.dim() == 4, "q must be 4D tensor");
    // ... 更多检查
    
    launch_flash_attention((float *)output.data_ptr(),
                           (const float *)q.data_ptr(),
                           (const float *)k.data_ptr(),
                           (const float *)v.data_ptr(),
                           static_cast<int>(batch_size),
                           static_cast<int>(num_heads),
                           static_cast<int>(seq_len),
                           static_cast<int>(head_dim),
                           scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_flash_attention",
          &torch_launch_flash_attention,
          "Flash Attention kernel wrapper");
}

