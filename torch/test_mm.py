import torch

def benchmark_torch_softmax(x, dim=-1, warmup=10, iters=50):
    # Warmup（避免第一次启动的开销）
    for _ in range(warmup):
        torch.nn.functional.softmax(x, dim=dim)
    torch.cuda.synchronize()

    # 使用 CUDA Event 测真正耗时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        torch.nn.functional.softmax(x, dim=dim)
    end.record()

    torch.cuda.synchronize()

    print("PyTorch softmax avg time:", start.elapsed_time(end) / iters, "ms")

x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
benchmark_torch_softmax(x)
