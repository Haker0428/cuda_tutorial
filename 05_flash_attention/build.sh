#!/bin/bash
# Flash Attention Extension 编译脚本

echo "开始编译 Flash Attention Extension..."

python setup.py build_ext --inplace

echo ""
echo "编译完成！可以使用以下命令测试："
echo "export LD_LIBRARY_PATH=/home/skyrain/miniconda3/lib/python3.10/site-packages/torch/lib:\$LD_LIBRARY_PATH"
echo "python test_flash_attention.py"

