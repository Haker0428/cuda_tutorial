import torch
import matadd
import time

def test_matadd():
    """测试矩阵加法功能"""
    print("测试 PyTorch Extension - 矩阵加法")
    print("=" * 50)
    
    # 测试参数
    n = 1024
    device = 'cuda'
    
    # 创建测试张量
    a = torch.randn(n, n, device=device, dtype=torch.float32)
    b = torch.randn(n, n, device=device, dtype=torch.float32)
    c = torch.zeros(n, n, device=device, dtype=torch.float32)
    
    # 使用 CUDA kernel
    print(f"矩阵大小: {n}x{n}")
    print(f"张量 a 形状: {a.shape}")
    print(f"张量 b 形状: {b.shape}")
    
    # 执行 CUDA kernel
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    matadd.torch_launch_matadd(c, a, b, n)
    end.record()
    
    torch.cuda.synchronize()
    kernel_time = start.elapsed_time(end)
    print(f"CUDA kernel 执行时间: {kernel_time:.4f} ms")
    
    # 使用 PyTorch 标准实现进行验证
    c_ref = a + b
    
    # 比较结果
    max_diff = torch.max(torch.abs(c - c_ref)).item()
    mean_diff = torch.mean(torch.abs(c - c_ref)).item()
    
    print(f"\n结果验证:")
    print(f"最大差异: {max_diff:.8f}")
    print(f"平均差异: {mean_diff:.8f}")
    
    if max_diff < 1e-5:
        print("✓ 测试通过！CUDA kernel 结果与 PyTorch 标准实现一致")
    else:
        print("✗ 测试失败！结果不匹配")
    
    # 性能对比
    torch.cuda.synchronize()
    start.record()
    c_pytorch = a + b
    end.record()
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end)
    
    print(f"\n性能对比:")
    print(f"PyTorch 标准实现: {pytorch_time:.4f} ms")
    print(f"CUDA kernel: {kernel_time:.4f} ms")
    if pytorch_time > 0:
        speedup = pytorch_time / kernel_time
        print(f"加速比: {speedup:.2f}x")
    
    return max_diff < 1e-5

if __name__ == "__main__":
    try:
        success = test_matadd()
        exit(0 if success else 1)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

