import torch
import triton
import triton.language as tl


# ---- 1) 定义 Triton kernel ----
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements,
                      BLOCK_SIZE: tl.constexpr):
    # 每个程序负责处理 BLOCK_SIZE 个元素
    pid = tl.program_id(axis=0)
    
    # 计算当前 program 处理的 index 区间
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # mask 用于处理越界情况
    mask = offsets < n_elements
    
    # 加载
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # 计算
    c = a + b
    
    # 写回
    tl.store(c_ptr + offsets, c, mask=mask)


# ---- 2) Python wrapper ----
def triton_vector_add(a, b):
    n = a.numel()
