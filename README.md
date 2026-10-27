# Transformer 中文笔记

<p align="center">
  <img src="https://img.shields.io/badge/articles-18-brightgreen" alt="articles published">
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
| 04 | [因果 mask 与自回归生成](articles/04-causal-mask-and-autoregressive-generation.md) | 训练为什么能并行、生成必须逐词？mask 如何保证\"看不见未来\"？ |
| 05 | [RoPE 的二维旋转推导与实现](articles/05-rope-rotation-derivation.md) | 旋转后的打分为什么只依赖相对位移？快慢频率各自负责什么？ |
| 06 | [残差连接、LayerNorm 与前馈网络](articles/06-residual-layernorm-ffn.md) | 梯度如何穿过深层网络？归一化与逐位置变换各自解决什么问题？ |
| 07 | [从 Encoder、Decoder 到完整 Transformer](articles/07-encoder-decoder-full-transformer.md) | Encoder 和 Decoder 如何协作完成序列到序列任务？交叉注意力的 Q、K、V 分别来自哪里？ |
| 08 | [KV Cache：推理时为什么能省计算](articles/08-kv-cache.md) | 生成时每一步在重复算什么？缓存 K、V 如何把每步计算从 O(t·d²) 降到 O(d²)，又付出了多少内存？ |
| 09 | [GQA/MQA：注意力头的参数共享](articles/09-gqa-mqa.md) | query 头数不变、只共享 K/V 头，为什么能省内存？压缩带来的秩约束代价是什么？ |
| 10 | [训练目标：交叉熵与 teacher forcing](articles/10-training-objective.md) | 模型到底在优化什么？训练喂真实词、推理喂自己词，这两者之间的鸿沟叫什么？ |
| 11 | [解码策略：temperature / top-p / top-k](articles/11-decoding-strategies.md) | 温度如何改变分布形态？top-k 与 top-p 截断的本质区别是什么？ |
| 12 | [Adam 优化器直觉：动量与自适应学习率](articles/12-adam-optimizer.md) | 动量如何平滑震荡并加速？自适应学习率怎样给每个参数定制步长？ |
| 13 | [学习率 warmup：为什么训练开头要用小学习率？](articles/13-learning-rate-warmup.md) | 训练初期梯度噪声大、Adam 第一步几乎只认方向不认大小，warmup 如何把学习率从 0 线性拉大来避开发散？ |
| 14 | [FlashAttention 直觉：IO 才是注意力的真正瓶颈](articles/14-flashattention.md) | 注意力的瓶颈为什么是显存读写而不是计算？分块和在线 softmax 如何在不存 N×N 矩阵的情况下得到完全相同的结果？ |
| 15 | [混合精度训练](articles/15-mixed-precision-training.md) | 为什么 fp16 算得快却不能直接拿来训练？主权重、loss scaling 如何把精度和速度都保住？ |
| 16 | [梯度检查点：用时间换显存](articles/16-gradient-checkpointing.md) | 反向传播为什么必须存下每层账本？只存段边界、需要时重算，如何把显存从 O(n) 压到 O(√n)？ |
| 17 | [RLHF 直觉：从人类偏好到奖励模型](articles/17-rlhf-preference-reward-model.md) | 人类只说\"这个回答更好\"、没有打分，奖励模型怎么从成对比较里学出一个标量分数？KL 惩罚又为什么能防止模型为了刷分走偏？ |
| 18 | [混合专家 MoE 直觉：只激活一部分参数，为什么反而更强？](articles/18-mixture-of-experts.md) | 只激活一两个专家，总参数变大、单步计算却不爆炸，代价是什么？ |

## 第一篇的阅读路线

位置与顺序 → 多频率正余弦 → Q/K 投影 → 打分的四项展开 → 相对位移性质 → softmax 权重 → 与 RoPE 的联系。

文章包含完整推导、数值示例、常见误区及自测题。想先建立直觉，可以先读第 1—3 节和第 7 节，再回到推导。

## 内容原则

- 先解释问题，再介绍公式与计算过程。
- 区分直觉、严格推导和需要实验验证的结论。
- 保持符号和向量方向一致。
- 优先引用原始论文；欢迎通过 Issue 指出错误或提出问题。
