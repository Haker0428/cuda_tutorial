from setuptools import setup
import os

# 延迟导入 torch，避免在构建依赖环境中找不到的问题
try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    raise ImportError(
        "PyTorch is required to build this extension. "
        "Please install PyTorch first: pip install torch"
    )

# 获取当前目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(current_dir, "include")

setup(
    name="flash_attention",
    version="0.1.0",
    include_dirs=[include_dir],
    ext_modules=[
        CUDAExtension(
            "flash_attention",
            [
                "aten/flash_attention_ops.cpp",
                "kernel/flash_attention.cu"
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
    zip_safe=False,
)

