# Transformer 中文笔记

<p align="center">
  <img src="https://img.shields.io/badge/articles-4-brightgreen" alt="articles published">
  <img src="https://img.shields.io/badge/new_article-daily-blue" alt="one new article every day">
  <img src="https://img.shields.io/badge/status-active-success" alt="series active">
</p>

用中文由浅入深讲解 Transformer 架构：从直觉出发，连接数学推导与实际计算，明确每个结论成立的条件。

适合希望理解注意力机制、位置编码及 Transformer 内部计算的读者。文章采用 GitHub Markdown，可直接阅读公式和示意图。

## 阅读目录

| 编号 | 文章 | 核心问题 |
|:-:|---|---|
| 01 | [正余弦位置编码如何影响注意力打分](articles/01-sinusoidal-positional-encoding-and-attention.md) | 位置信息如何进入 Q、K？相对位移性质为什么不等于纯相对位置打分？ |
| 02 | [缩放点积注意力：为什么除以 √d_k](articles/02-scaled-dot-product-attention.md) | 点积的方差为什么是 d_k？缩放如何影响 softmax 形态与梯度？ |
| 03 | [多头注意力：不同的头究竟在学习什么？](articles/03-multi-head-attention.md) | 不同的头为什么能学到不同的东西？多头注意力的参数量真的变多了吗？ |
| 04 | [因果 mask 与自回归生成](articles/04-causal-mask-and-autoregressive-generation.md) | 训练为什么能并行、生成必须逐词？mask 如何保证"看不见未来"？ |

## 第一篇的阅读路线

位置与顺序 → 多频率正余弦 → Q/K 投影 → 打分的四项展开 → 相对位移性质 → softmax 权重 → 与 RoPE 的联系。

文章包含完整推导、数值示例、常见误区及自测题。想先建立直觉，可以先读第 1—3 节和第 7 节，再回到推导。

## 后续选题方向

以下为选题方向，尚未发布：

- RoPE 的二维旋转推导与实现。
- 残差连接、LayerNorm 与前馈网络。
- 从 Encoder、Decoder 到完整 Transformer。

## 内容原则

- 先解释问题，再介绍公式与计算过程。
- 区分直觉、严格推导和需要实验验证的结论。
- 保持符号和向量方向一致。
- 优先引用原始论文；欢迎通过 Issue 指出错误或提出问题。
