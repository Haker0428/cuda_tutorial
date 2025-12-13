# PyTorch CUDA Extension - 矩阵加法

这是一个 PyTorch CUDA Extension 示例，实现了矩阵加法的 CUDA kernel。

## 项目结构

```
03_torch_extenssion/
├── kernel/
│   └── matadd.cu          # CUDA kernel 实现
├── include/
│   └── matadd.h           # CUDA kernel 头文件
├── aten/
│   └── mat_add_ops.cpp    # PyTorch 绑定代码
├── setup.py               # 构建脚本
├── test_matadd.py         # 测试脚本
└── README.md              # 本文档
```

## 构建

1. 确保已安装 PyTorch 和 CUDA toolkit

2. 运行构建命令（推荐方式）：

**方式 1：使用 build_ext --inplace（推荐，最简单）**
```bash
python setup.py build_ext --inplace
```
这会在当前目录生成 `.so` 文件，可以直接导入使用。

**注意**：使用前需要设置环境变量：
```bash
export LD_LIBRARY_PATH=/home/skyrain/miniconda3/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```
或者添加到 `~/.bashrc` 中永久生效。

**方式 2：使用 pip 安装（需要 pyproject.toml）**
```bash
pip install -e .
```

**方式 3：使用提供的编译脚本**
```bash
bash build.sh
```

> **注意**：对于 PyTorch CUDA Extension，推荐使用 `python setup.py build_ext --inplace`，这是最简单直接的方式，会生成编译好的扩展模块在当前目录。

## 使用示例

```python
import torch
import matadd

n = 1024
a = torch.randn(n, n, device='cuda', dtype=torch.float32)
b = torch.randn(n, n, device='cuda', dtype=torch.float32)
c = torch.zeros(n, n, device='cuda', dtype=torch.float32)

# 调用 CUDA kernel
matadd.torch_launch_matadd(c, a, b, n)

# 验证结果
c_ref = a + b
print(torch.allclose(c, c_ref))
```

## 测试

运行测试脚本：

```bash
python test_matadd.py
```

测试脚本会：
- 验证 CUDA kernel 结果的正确性
- 对比性能与 PyTorch 标准实现

## 实现细节

- **CUDA Kernel**: 使用 16x16 的 thread block 大小
- **Grid 配置**: 自动计算 grid 大小以覆盖所有矩阵元素
- **错误检查**: 包含 CUDA 错误检查和同步

# 编译说明

## 快速编译（推荐）

```bash
cd /home/skyrain/Cuda/cuda_tutorial/03_torch_extenssion
python setup.py build_ext --inplace
```

## 使用编译好的模块

**重要**：需要设置 LD_LIBRARY_PATH 才能导入模块：

```bash
export LD_LIBRARY_PATH=/home/skyrain/miniconda3/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
python test_matadd.py
```

或者将上述 export 命令添加到 `~/.bashrc` 中永久生效。

## 测试结果

- ✅ 编译成功
- ✅ 模块导入成功
- ✅ 功能验证通过（与 PyTorch 标准实现结果一致）
- ✅ 性能提升：CUDA kernel 比 PyTorch 标准实现快 **1.79倍**

## 生成的文件

编译后会生成：
- `matadd.cpython-310-aarch64-linux-gnu.so` - 编译好的扩展模块

