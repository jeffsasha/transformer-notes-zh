# Transformer 中文笔记

<p align="center">
  <img src="https://img.shields.io/badge/articles-32-brightgreen" alt="articles published">
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
| 19 | [DPO 直觉：不用奖励模型，怎么直接从偏好学对齐？](articles/19-dpo-direct-preference-optimization.md) | 奖励模型和 PPO 都可以省掉吗？最优策略本身如何反解出奖励？ |
| 20 | [LoRA 微调直觉：低秩更新为什么够用？](articles/20-lora-low-rank-adaptation.md) | 只训 $r(d+k)$ 个参数，为什么能逼近全参数微调？低秩假设的边界在哪里？ |
| 21 | [量化直觉：把权重压到 8-bit/4-bit，为什么模型还没坏？](articles/21-quantization-intuition.md) | int8/int4 的舍入误差上限是多少？为什么误差没把模型压垮？ |
| 22 | [权重衰减与 AdamW：为什么 Adam 加 L2 不等于 AdamW？](articles/22-weight-decay-and-adamw.md) | SGD 里 L2 等于权重衰减，为什么到了 Adam 里这个等式会失效？ |
| 23 | [学习率调度：warmup 之后，为什么还要让学习率衰减？](articles/23-learning-rate-schedule.md) | 恒定学习率为什么会在最优解附近打转？衰减如何把噪声球越缩越小，又为什么要在总步数确定后才谈衰减？ |
| 24 | [Scaling Law 直觉：参数、数据、算力怎么分配？](articles/24-scaling-law.md) | 算力约束 C ≈ 6ND 是怎么来的？固定算力下参数量 N 和数据量 D 为什么要按比例一起增长？ |
| 25 | [梯度裁剪：为什么大梯度会炸掉训练？](articles/25-gradient-clipping.md) | 梯度范数过大时，一步更新为什么会把参数推出山谷？裁剪只缩短步长、不改变方向，又为什么不能掩盖真正的问题？ |
| 26 | [位置外推：训练短、推理长时为什么会崩？](articles/26-length-extrapolation.md) | 序列变长时，softmax 分母、RoPE 旋转角和位置向量各出了什么问题？ |
| 27 | [注意力 sink：为什么开头几个 token 总吸走注意力？](articles/27-attention-sink.md) | softmax 的权重和必须为 1，模型把多余的注意力“倒”去了哪里？为什么删掉开头几个 token，模型就崩了？ |
| 28 | [权重初始化：为什么深层网络一开始就要把方差算对？](articles/28-weight-initialization.md) | 初始化标准差差一点点，十层之后信号就缩水百万倍或爆炸亿倍，方差该按什么公式算？ |
| 29 | [Dropout 直觉：训练时随机丢掉神经元，为什么反而更不容易过拟合？](articles/29-dropout.md) | 训练时随机丢掉一部分神经元，测试时全开——期望为什么能对上？随机性怎么变成正则化？ |
| 30 | [Batch Size 与学习率：大 batch 为什么要配大学习率？](articles/30-batch-size-and-learning-rate.md) | batch 翻倍时学习率为什么也要翻倍？超过临界 batch size 后再加 batch 为什么只剩浪费算力？ |
| 31 | [数据并行与张量并行](articles/31-data-parallelism-and-tensor-parallelism.md) | 数据、参数、梯度三种切分方式各解决什么问题？all-reduce 的通信量为什么是 2S？ |
| 32 | [ZeRO：单卡显存怎么从 16Ψ 降到 16Ψ/N？](articles/32-zero-redundancy-optimizer.md) | 数据并行每张卡都存着 3 份相同的模型状态，ZeRO 把参数、梯度和优化器状态切开后，显存降了多少、通信又多了多少？ |

## 第一篇的阅读路线

位置与顺序 → 多频率正余弦 → Q/K 投影 → 打分的四项展开 → 相对位移性质 → softmax 权重 → 与 RoPE 的联系。

文章包含完整推导、数值示例、常见误区及自测题。想先建立直觉，可以先读第 1—3 节和第 7 节，再回到推导。

## 内容原则

- 先解释问题，再介绍公式与计算过程。
- 区分直觉、严格推导和需要实验验证的结论。
- 保持符号和向量方向一致。
- 优先引用原始论文；欢迎通过 Issue 指出错误或提出问题。
