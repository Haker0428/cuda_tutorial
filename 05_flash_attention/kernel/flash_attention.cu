#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cmath>

#ifndef BLOCK_SIZE_K
#define BLOCK_SIZE_K 64  // K/V 的块大小，可根据硬件调整
#endif

// ============================================================================
// 关键步骤封装：方便未来优化
// ============================================================================

/**
 * 步骤1: 计算 Q @ K^T 分数并找到 block 内的最大值
 * 
 * @param q_off Q 向量的偏移量
 * @param k_off K 向量的偏移量
 * @param q Q 矩阵指针
 * @param k K 矩阵指针
 * @param scale 缩放因子 (1/sqrt(head_dim))
 * @param scores 输出的分数数组
 * @param k_start K block 的起始索引
 * @param k_end K block 的结束索引
 * @return block 内的最大分数值
 */
template<int HEAD_DIM>
__device__ __forceinline__ float compute_qk_scores_and_max(
    const int q_off,
    const int base,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float scale,
    float* scores,
    const int k_start,
    const int k_end)
{
    float m_block = -INFINITY;
    
    for (int kj = k_start; kj < k_end; ++kj) {
        const int k_off = base + kj * HEAD_DIM;
        
        // 计算 Q @ K^T (点积)
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot = fmaf(q[q_off + d], k[k_off + d], dot);
        }
        
        // 应用缩放
        float score = dot * scale;
        scores[kj - k_start] = score;
        m_block = fmaxf(m_block, score);
    }
    
    return m_block;
}

/**
 * 步骤2: 计算 block 内的 softmax 和加权 V
 * 
 * @param scores 分数数组
 * @param v V 矩阵指针
 * @param base V 的基础偏移量
 * @param m_block block 内的最大值（用于数值稳定）
 * @param k_start K block 的起始索引
 * @param k_end K block 的结束索引
 * @param l_block 输出的 softmax 分母（归一化因子）
 * @param o_block 输出的加权 V 向量（分子）
 */
template<int HEAD_DIM>
__device__ __forceinline__ void compute_block_softmax_and_weighted_v(
    const float* scores,
    const float* __restrict__ v,
    const int base,
    const float m_block,
    const int k_start,
    const int k_end,
    float& l_block,
    float* o_block)
{
    l_block = 0.0f;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) o_block[d] = 0.0f;
    
    for (int kj = k_start; kj < k_end; ++kj) {
        float score = scores[kj - k_start];
        
        // 数值稳定的 exp: exp(score - m_block)
        float prob = __expf(score - m_block);
        l_block += prob;
        
        // 累加加权 V: prob * V[kj]
        const int v_off = base + kj * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            o_block[d] = fmaf(prob, v[v_off + d], o_block[d]);
        }
    }
}

/**
 * 步骤3: 合并 block 统计信息到运行状态（Online Softmax 的核心）
 * 
 * 这是 Flash Attention 的关键步骤：将新 block 的结果与之前累积的结果合并。
 * 通过缩放因子确保数值稳定性。
 * 
 * @param m_old 之前的运行最大值
 * @param m_block 当前 block 的最大值
 * @param l_old 之前的运行分母
 * @param l_block 当前 block 的分母
 * @param o_old 之前的运行输出向量（分子）
 * @param o_block 当前 block 的输出向量（分子）
 * @param m_new 输出的新运行最大值
 * @param l_new 输出的新运行分母
 */
template<int HEAD_DIM>
__device__ __forceinline__ void merge_block_stats(
    const float m_old,
    const float m_block,
    const float l_old,
    const float l_block,
    const float* o_old,
    const float* o_block,
    float& m_new,
    float& l_new,
    float* o_new)
{
    // 计算新的最大值
    m_new = fmaxf(m_old, m_block);
    
    // 计算缩放因子（用于将旧值缩放到新的数值范围）
    float alpha_old = __expf(m_old - m_new);    // <= 1.0
    float alpha_block = __expf(m_block - m_new); // <= 1.0
    
    // 合并分母: l_new = l_old * alpha_old + l_block * alpha_block
    l_new = l_old * alpha_old + l_block * alpha_block;
    
    // 合并分子（输出向量）: o_new = o_old * alpha_old + o_block * alpha_block
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        o_new[d] = o_old[d] * alpha_old + o_block[d] * alpha_block;
    }
}

/**
 * 步骤4: 归一化输出
 * 
 * @param o 累积的输出向量（分子）
 * @param l 累积的分母
 * @param out 输出的归一化结果
 * @param out_off 输出向量的偏移量
 */
template<int HEAD_DIM>
__device__ __forceinline__ void normalize_output(
    const float* o,
    const float l,
    float* __restrict__ out,
    const int out_off)
{
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        out[out_off + d] = o[d] * inv_l;
    }
}

// ============================================================================
// 主 Kernel 函数
// ============================================================================

template<int HEAD_DIM>
__global__ void FlashAttnRefKernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int seq_len,  // 只保留 seq_len，batch_size 和 num_heads 从 gridDim 获取
    float scale)
{
    // 从 grid 维度获取 batch_size 和 num_heads
    const int batch_size = gridDim.z;
    const int num_heads = gridDim.y;
    
    // 计算当前线程处理的 (batch, head, query_idx)
    int b = blockIdx.z;
    int h = blockIdx.y;
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size || h >= num_heads || qi >= seq_len) return;

    // 计算内存偏移量
    const int stride_bh = num_heads * seq_len * HEAD_DIM;
    const int base = b * stride_bh + h * (seq_len * HEAD_DIM);
    const int q_off = base + qi * HEAD_DIM;

    // 初始化运行状态（Online Softmax 状态）
    float m = -INFINITY;  // running max
    float l = 0.0f;        // running sum (分母)
    float o[HEAD_DIM];     // running output (分子)
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) o[d] = 0.0f;

    // 分块处理 K 和 V
    const int nblocks = (seq_len + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;

    for (int bk = 0; bk < nblocks; ++bk) {
        int k_start = bk * BLOCK_SIZE_K;
        int k_end = min(k_start + BLOCK_SIZE_K, seq_len);

        // 临时存储
        float scores[BLOCK_SIZE_K];
        #pragma unroll
        for (int t = 0; t < BLOCK_SIZE_K; ++t) scores[t] = -INFINITY;

        // ============================================================
        // 步骤1: 计算 Q @ K^T 分数并找到 block 内的最大值
        // ============================================================
        float m_block = compute_qk_scores_and_max<HEAD_DIM>(
            q_off, base, q, k, scale, scores, k_start, k_end);

        // ============================================================
        // 步骤2: 计算 block 内的 softmax 和加权 V
        // ============================================================
        float l_block = 0.0f;
        float o_block[HEAD_DIM];
        compute_block_softmax_and_weighted_v<HEAD_DIM>(
            scores, v, base, m_block, k_start, k_end, l_block, o_block);

        // ============================================================
        // 步骤3: 合并 block 统计信息到运行状态（Online Softmax 核心）
        // ============================================================
        float m_new, l_new;
        float o_new[HEAD_DIM];
        merge_block_stats<HEAD_DIM>(
            m, m_block, l, l_block, o, o_block, m_new, l_new, o_new);

        // 更新运行状态
        m = m_new;
        l = l_new;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            o[d] = o_new[d];
        }
    }

    // ============================================================
    // 步骤4: 归一化输出
    // ============================================================
    const int out_off = base + qi * HEAD_DIM;
    normalize_output<HEAD_DIM>(o, l, out, out_off);
}

// dispatch: only common head_dim
void launch_flash_attention(
    float* out, const float* q, const float* k, const float* v,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale)
{
    dim3 block(128);
    dim3 grid((seq_len + block.x - 1) / block.x, num_heads, batch_size);

    if (head_dim == 32) {
        FlashAttnRefKernel<32><<<grid, block>>>(out, q, k, v, seq_len, scale);
    } else if (head_dim == 64) {
        FlashAttnRefKernel<64><<<grid, block>>>(out, q, k, v, seq_len, scale);
    } else if (head_dim == 128) {
        FlashAttnRefKernel<128><<<grid, block>>>(out, q, k, v, seq_len, scale);
    } else {
        fprintf(stderr, "Unsupported head_dim=%d. Use 32/64/128 for this ref kernel.\n", head_dim);
        return;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA launch error: %s\n", cudaGetErrorString(err));
    }
}
