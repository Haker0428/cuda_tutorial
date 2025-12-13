#!/bin/bash
# PyTorch CUDA Extension 编译脚本

echo "开始编译 PyTorch CUDA Extension..."

# 方式 1: 使用 python setup.py build_ext --inplace (推荐，最简单)
echo "使用方式 1: python setup.py build_ext --inplace"
python setup.py build_ext --inplace

echo ""
echo "编译完成！可以使用以下命令测试："
echo "python test_matadd.py"

