[TOC]

# 传统Attenion分析

$$
\mathrm{Attention}(Q,K,V)
=
\mathrm{Softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right)V
$$



假设sequence length = $N$ , hidden dim = $d$  

* $QK⊤$: $O(N^2d)$计算
* $Softmax$：$O(N^2)$
* 乘 $V$：$O(N^2 d)$



## 📌 IO分析

传统Attention必须构建整个$score$ 矩阵：
$$
S=QK^{\top}
$$
我们假设sequence length为$N=4096$，$head$  维度$d=128$， 并且我们假设数据类型为FP16， 一步步拆解IO：



### **读取$Q / K$**

Q 大小：4096 × 128 × 2B = **1,048,576 B** ≈ **1 MB**
	K 大小：同 Q = **1 MB**



###  计算 QKᵀ → 写出 score 矩阵

S 大小：
$$
4096 \times 4096 \times 2B = 33,554,432 B = 32 MB
$$
这一步 IO：

- 写出 S（32 MB）
- Softmax 再读 S（32 MB）
- Softmax 写出概率矩阵 P（32 MB）
- 再读 P（32 MB）去乘 V

**光中间矩阵读写就超过 128 MB**





### **传统 Attention 总 IO 粗略统计：**

| 操作              | IO        |
| ----------------- | --------- |
| 读 Q              | **1 MB**  |
| 读 K              | **1 MB**  |
| 写 Score 矩阵 S   | **32 MB** |
| 读 Score 矩阵 S   | **32 MB** |
| 写 Softmax 结果 P | **32 MB** |
| 读 Softmax 结果 P | **32 MB** |
| 读 V              | **1 MB**  |

总计 IO ≈ **131 MB**

> [!IMPORTANT]
>
> **核心瓶颈：反复读写两个巨大的矩阵 S、P（每个 4096×4096）**



# Attention的变形

##   Softmax的向量化的表示

我们先来拆解一下普通 softmax：
$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$
数值稳定写法：
$$
\frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m=\max_j x_j
$$


如果是以一个query $i$ 作为视角，标准 $Attention$ 定义是：
$$
\text{Attention}(Q,K,V)_i
= \sum_{j=1}^{N_k} \text{softmax}(S_{i,j}) \cdot V_j
$$


> [!CAUTION]
>
> 我们可以注意到一点，**Softmax 是“按行”做的**

因此
$$
m_i = \max_j S_{i,j}
$$
**max 的维度是：`Nk（key/token 维度）`**



对某一个固定的 query $i$，softmax 是：
$$
\text{softmax}(S_{i,j})
=
\frac{e^{S_{i,j}}}{\sum\limits_{k=1}^{N_k} e^{S_{i,k}}}
$$
输出是：

对一个 query $i$：
$$
O_i = \sum_j \frac{e^{S_{i,j}}}{\sum_k e^{S_{i,k}}} \; V_j
$$
把它改写成：
$$
O_i = \frac{\sum_j e^{S_{i,j}} V_j}{\sum_k e^{S_{i,k}}}
$$
**注意这个结构**：

- 你真正需要的，只有两样：
  1. $\sum_j e^{S_{i,j}} V_j$（一个向量）
  2. $\sum_k e^{S_{i,k}}$（一个标量）

## 指数变换



看任意一个旧项 $k \le t$ 的权重：

旧坐标系里的“分子项”是：
$$
e^{x_k - m^{(t)}}
$$
新坐标系下同一个项应该变成：
$$
e^{x_k - m^{(t+1)}}
$$
两者关系：
$$
e^{x_k - m^{(t+1)}} 
= e^{x_k - m^{(t)}} \cdot e^{m^{(t)} - m^{(t+1)}}
$$
因为：
$$
x_k - m^{(t+1)} = (x_k - m^{(t)}) + (m^{(t)} - m^{(t+1)})
$$

### softmax分母项

把上面的关系对所有旧项求和：
$$
\sum_{k=1}^{t} e^{x_k - m^{(t+1)}} 
= \sum_{k=1}^{t} \Big(e^{x_k - m^{(t)}} \cdot e^{m^{(t)} - m^{(t+1)}}\Big)
$$
把常数提出去：
$$
= e^{m^{(t)} - m^{(t+1)}} \sum_{k=1}^{t} e^{x_k - m^{(t)}}
$$

### softmax分子项

同理：
$$
\sum_{k=1}^{t} e^{x_k - m^{(t+1)}} V_k
= \sum_{k=1}^{t} \Big(e^{x_k - m^{(t)}} \cdot e^{m^{(t)} - m^{(t+1)}}\Big) V_k
$$
依旧把常数提出去：
$$
= e^{m^{(t)} - m^{(t+1)}} \sum_{k=1}^{t} e^{x_k - m^{(t)}} V_k
$$

# FlashAttention浅析

<img src="./assets/flashattention.png" alt="https://miro.medium.com/1%2AmghON4aLZfqb9oMMioZiMg.png?utm_source=chatgpt.com" style="zoom:67%;" />



## 🚩方案分析 

从上边推导可知：
$$
\mathrm{Attention}(Q,K,V)
=
\mathrm{Softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right)V
$$
我们从上述公式转换为了下述形式：
$$
O_i = \frac{\sum_j e^{S_{i,j}} V_j}{\sum_k e^{S_{i,k}}}
$$

###  分块逻辑浅析

在原有的传统Attention计算中，对于每个query必须计算完一整行的信息后，使用softmax进行注意力权重矩阵的计算，因此需要存储完整的$S$矩阵，再加上来回的写入写出到时候了memory-bound，因此Flash Attention的方法提出了一种新的方式计算，名为Online-SoftMax。

我们假设维度信息如下

```
Q        : [Nq, D]
K^T      : [D, Nk]
S = QK^T : [Nq, Nk]
```

然后经过分块矩阵乘得到的$S$  如下图，

```
           K tokens →
        k0   k1   k2   k3   k4   ...
q0     s00  s01  s02  s03  s04
q1     s10  s11  s12  s13  s14
q2     s20  s21  s22  s23  s24
...
```

FlashAttention的采取的方法是，通过记录中间累加值，通过迭代的方式，不断朝$N_k$ 方向进行计算。

从分块的角度来看：

```
s00 -> s01 -> s02 -> s03
```

> [!IMPORTANT]
>
> **分块之间必须保持串行，分块内可以充分并行**
>
> **Nk是时间轴**

这个是分块逻辑，我们先有一个初步的印象。

那就会有问题了，对于softmax而言，他的不断累积性导致了我们必须完整一整行才能计算softmax，online-softmax是怎么做到的？

### On-line softmax

$$
O_i = \frac{\sum_j e^{S_{i,j}-m(t)} V_j}{\sum_k e^{S_{i,k}-m(t)}}
$$

我们再来分析拆解后的 $softmax$ 公式：

1. $\sum_j e^{S_{i,j}-m(t)} V_j$（一个向量）
2. $\sum_k e^{S_{i,k}-m(t)}$（一个标量）

其中$m(t)$ 代表计算过的累积分块内的局部最大值。

从分块的串行顺序决定了累加顺序，并且串行的约束为公式引入了时间步状态，因此，我们只用维护累加值和最大值，而累加值分为了分子和分母。

它维护的是**在线状态**：

- $m$：目前看过的列里的最大值（running max）
- $l$：在对应最大值归一化下的指数和（running sum）（分母）
- $O$：指数加权的 $V$ 累积（running output）（分子）

每处理完一个 K block，会得到该 block 的局部统计：

- $m_b$：该 block 内这一行的最大值
- $l_b=\sum_{j\in block} e^{S_{i,j}-m_b}$
- $O_b=\sum_{j\in block} e^{S_{i,j}-m_b} V_j$

然后每次处理一个基本块后，一旦发现更大的max值，就会对在线更新这些值 ，将旧累积计整体缩放到新的值域空间：

然后做一次 **merge**：
$$
\begin{aligned}
m_{\text{new}} &= \max(m_{\text{old}}, m_b)\\
l_{\text{new}} &= l_{\text{old}} \cdot e^{m_{\text{old}}-m_{\text{new}}}
               + l_b \cdot e^{m_b-m_{\text{new}}}\\
O_{\text{new}} &= O_{\text{old}} \cdot e^{m_{\text{old}}-m_{\text{new}}}
               + O_b \cdot e^{m_b-m_{\text{new}}}
\end{aligned}
$$

### 块内逻辑

