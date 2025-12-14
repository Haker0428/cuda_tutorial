import torch
import torch.nn.functional as F
import math

# 尝试导入 flash_attention 模块
try:
    import flash_attention
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("注意: flash_attention 模块未找到，将只测试传统 Attention 性能")

def standard_attention(q, k, v, scale):
    """标准注意力机制实现（用于对比）"""
    # Q @ K^T * scale (scale = 1/sqrt(head_dim))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    # Softmax
    attn_weights = F.softmax(scores, dim=-1)
    # @ V
    output = torch.matmul(attn_weights, v)
    return output

def test_flash_attention():
    """测试 Flash Attention 功能"""
    print("测试 PyTorch Extension - Flash Attention")
    print("=" * 50)
    
    # TODO: 设置测试参数
    batch_size = 2
    num_heads = 8
    seq_len = 512
    head_dim = 64
    device = 'cuda'
    
    scale = 1.0 / math.sqrt(head_dim)
    
    # TODO: 创建测试张量
    # 形状: (batch_size, num_heads, seq_len, head_dim)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    output = torch.zeros_like(q)
    
    print(f"Batch size: {batch_size}")
    print(f"Num heads: {num_heads}")
    print(f"Seq len: {seq_len}")
    print(f"Head dim: {head_dim}")
    print(f"Scale: {scale:.6f}")
    
    if not FLASH_ATTENTION_AVAILABLE:
        print("Flash Attention 模块未找到，请先编译扩展")
        return
    
    # 执行 CUDA kernel
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    flash_attention.torch_launch_flash_attention(
        output, q, k, v, batch_size, num_heads, seq_len, head_dim, scale
    )
    end.record()
    
    torch.cuda.synchronize()
    kernel_time = start.elapsed_time(end)
    print(f"CUDA kernel 执行时间: {kernel_time:.4f} ms")
    
    # 使用标准注意力机制进行验证
    output_ref = standard_attention(q, k, v, scale)
    
    # 比较结果
    max_diff = torch.max(torch.abs(output - output_ref)).item()
    mean_diff = torch.mean(torch.abs(output - output_ref)).item()
    
    print(f"\n结果验证:")
    print(f"最大差异: {max_diff:.8f}")
    print(f"平均差异: {mean_diff:.8f}")
    
    tolerance = 1e-3  # Flash Attention 可能精度稍低
    if max_diff < tolerance:
        print(f"✓ 测试通过！")
    else:
        print(f"✗ 测试失败！")
    
    # 性能对比
    print(f"\n性能对比:")
    torch.cuda.synchronize()
    start.record()
    for _ in range(10):
        _ = standard_attention(q, k, v, scale)
    end.record()
    torch.cuda.synchronize()
    standard_time = start.elapsed_time(end) / 10.0
    print(f"标准 Attention 平均时间: {standard_time:.4f} ms")
    print(f"Flash Attention 时间: {kernel_time:.4f} ms")
    if kernel_time > 0:
        print(f"加速比: {standard_time / kernel_time:.2f}x")

def benchmark_attention_performance():
    """性能对比测试：传统 Attention vs Flash Attention"""
    print("\n" + "=" * 50)
    print("性能对比测试：传统 Attention vs Flash Attention")
    print("=" * 50)
    
    # 测试配置
    test_configs = [
        {"batch_size": 1, "num_heads": 8, "seq_len": 512, "head_dim": 64, "name": "小规模"},
        {"batch_size": 2, "num_heads": 8, "seq_len": 1024, "head_dim": 64, "name": "中等规模"},
        # {"batch_size": 4, "num_heads": 8, "seq_len": 2048, "head_dim": 64, "name": "大规模"},
        # {"batch_size": 2, "num_heads": 16, "seq_len": 4096, "head_dim": 64, "name": "超大规模"},
    ]
    
    device = 'cuda'
    warmup_iterations = 10
    benchmark_iterations = 50
    
    results = []
    
    for config in test_configs:
        batch_size = config["batch_size"]
        num_heads = config["num_heads"]
        seq_len = config["seq_len"]
        head_dim = config["head_dim"]
        scale = 1.0 / math.sqrt(head_dim)
        
        print(f"\n配置: {config['name']} (batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim})")
        print("-" * 50)
        
        # 创建测试张量
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
        
        # Warmup
        for _ in range(warmup_iterations):
            _ = standard_attention(q, k, v, scale)
        torch.cuda.synchronize()
        
        # 测试传统 Attention
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(benchmark_iterations):
            _ = standard_attention(q, k, v, scale)
        end.record()
        torch.cuda.synchronize()
        standard_time = start.elapsed_time(end) / benchmark_iterations
        
        print(f"传统 Attention 平均时间: {standard_time:.4f} ms")
        
        # 测试 Flash Attention (如果已实现)
        if FLASH_ATTENTION_AVAILABLE:
            try:
                output = torch.zeros_like(q)
                
                # Warmup
                for _ in range(warmup_iterations):
                    flash_attention.torch_launch_flash_attention(
                        output, q, k, v, batch_size, num_heads, seq_len, head_dim, scale
                    )
                torch.cuda.synchronize()
                
                start.record()
                for _ in range(benchmark_iterations):
                    flash_attention.torch_launch_flash_attention(
                        output, q, k, v, batch_size, num_heads, seq_len, head_dim, scale
                    )
                end.record()
                torch.cuda.synchronize()
                flash_time = start.elapsed_time(end) / benchmark_iterations
                
                speedup = standard_time / flash_time
                print(f"Flash Attention 平均时间: {flash_time:.4f} ms")
                print(f"加速比: {speedup:.2f}x")
                
                results.append({
                    "config": config["name"],
                    "standard_time": standard_time,
                    "flash_time": flash_time,
                    "speedup": speedup
                })
            except Exception as e:
                print(f"Flash Attention 执行出错: {e}")
                results.append({
                    "config": config["name"],
                    "standard_time": standard_time,
                    "flash_time": None,
                    "speedup": None
                })
        else:
            print("Flash Attention 未实现，跳过测试")
            results.append({
                "config": config["name"],
                "standard_time": standard_time,
                "flash_time": None,
                "speedup": None
            })
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("性能测试汇总")
    print("=" * 50)
    print(f"{'配置':<15} {'传统Attention(ms)':<20} {'FlashAttention(ms)':<20} {'加速比':<10}")
    print("-" * 70)
    for result in results:
        flash_str = f"{result['flash_time']:.4f}" if result['flash_time'] else "N/A"
        speedup_str = f"{result['speedup']:.2f}x" if result['speedup'] else "N/A"
        print(f"{result['config']:<15} {result['standard_time']:<20.4f} {flash_str:<20} {speedup_str:<10}")
    
    return results

def test_flash_attention_different_sizes():
    """测试不同大小的 Flash Attention"""
    print("\n" + "=" * 50)
    print("测试不同大小的 Flash Attention")
    print("=" * 50)
    
    # TODO: 实现不同大小的测试
    pass

if __name__ == "__main__":
    try:
        test_flash_attention()
        # test_flash_attention_different_sizes()
        if FLASH_ATTENTION_AVAILABLE:
            benchmark_attention_performance()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

