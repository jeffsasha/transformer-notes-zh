# 07｜从 Encoder、Decoder 到完整 Transformer

> 阅读目标：理解完整的 Transformer 如何组织 Encoder 和 Decoder 协作完成序列到序列任务；掌握交叉注意力（cross-attention）中 Q、K、V 的来源与维度；能从计算图上解释训练时并行、生成时串行的数学原因。
>
> 前置知识：第 02 篇缩放点积注意力公式；第 04 篇因果 mask 与自回归生成；第 06 篇 Transformer block 的内部结构（残差、LayerNorm、FFN）。

前六篇把 Transformer 的零件逐个拆透了：位置编码、注意力、多头、mask、残差、归一化、前馈网络。这一篇把它们全部拼起来，看原论文（Vaswani et al., 2017）的完整结构如何工作。

> 结论性质声明：本篇中"维度推导""因果 mask 下的依赖关系""参数量计数"是**严格推导**；"Encoder 适合理解、Decoder 适合生成""哪种结构在某任务上更好"是**需实验验证**的结论，文中逐一标注。

## 1. 问题：为什么要有 Encoder 和 Decoder 两套结构？

**直觉。** 拿机器翻译举例——把"我爱猫"翻成 "I love cats"，这个任务天然分成两步：

1. 先**读懂**整句源语言：读到"猫"时，你的眼睛可以扫回整句确认它是宾语，做阅读理解时没人只许你往前看；
2. 再**逐词写出**目标语言：写第 3 个词时，你只能依赖已经写出的前 2 个词，不可能"看到"还没写出的未来。

一步"全向阅读"加一步"单向续写"——Encoder 管前者，Decoder 管后者。这就是 Encoder-Decoder 结构的任务动机。

**严格表述。** Encoder 把源序列 $X = [x_1, \dots, x_n]$ 编码成一组上下文表示 $Z = [z_1, \dots, z_n]$；Decoder 以 $Z$ 为条件，自回归地生成目标序列 $Y = [y_1, \dots, y_m]$，即建模条件概率 $P(y_t \mid y_{<t}, Z)$。注意 $n$ 和 $m$ 可以不等（"我爱猫" 3 个字，"I love cats" 3 个词，德语翻译可能更长）——这正是需要两套结构的根本原因：输入输出是**两个长度未必相等的序列**。

## 2. Encoder：整句一起读的"阅读理解器"

**直觉。** 做阅读理解时，你的眼睛可以在整句话上来回扫：读到代词"它"时回头找"猫"。Encoder 的自注意力**不加因果 mask**，每个位置都能 attend 到所有位置，包括它后面的词。

**严格推导。** Encoder 第 $l$ 层的自注意力（单头简化版），输入 $H^{(l-1)} \in \mathbb{R}^{n \times d}$：

```math
\mathrm{Attn}(H) = \mathrm{softmax}\!\left(\frac{(HW_Q)(HW_K)^\top}{\sqrt{d_k}}\right)(HW_V)
```

无 mask 意味着 softmax 的每一行都是对**全部 $n$ 个位置**的加权平均——位置 $i$ 的输出是所有位置 value 的凸组合。堆叠 $N$ 层之后，每个 $z_i$ 都融合了整句的信息（严格的说法是：信息可以通过多跳路径从任意位置 $j$ 传到位置 $i$；"融合得好不好、用到了多少"是实验问题）。

一个 Encoder block 就是第 06 篇讲过的 block（自注意力 + 残差 + LayerNorm + FFN），堆 $N$ 层：

```math
Z = \mathrm{Encoder}(X), \qquad Z \in \mathbb{R}^{n \times d}
```

**维度守恒**（严格）：输入 $X \in \mathbb{R}^{n \times d}$，每一层的输出都保持 $n \times d$（第 06 篇逐项核对过），所以 $Z$ 也是 $n \times d$——位置数 $n$ 不变，每个位置还是 $d$ 维。

**位置信息的来源**（严格）：Encoder 自身不感知顺序——顺序完全来自第 01/05 篇的位置编码。无位置编码时，自注意力是**置换等变**的：交换输入中两个位置，输出的对应两个位置也原样交换，模型根本察觉不到顺序变化。所以"Encoder 能利用词序"成立的条件是：输入已经叠加了位置编码。

## 3. Decoder：只能往前看的"续写机"

**直觉。** 写文章时你看不到还没写出的下一句。Decoder 每一层有两个注意力子层，各管一件事：

1. **带因果 mask 的自注意力**：位置 $t$ 只能看 $y_{\le t}$——"我已经写了什么"（第 04 篇已讲透：训练时加 mask 一次算出全部位置，数学上等价于逐词生成）；
2. **交叉注意力（cross-attention）**：每写一个词都"回头查字典"——去看 Encoder 输出的整句源语言表示，"原文对应的是什么"。

**严格推导（本篇核心公式）。** 设 Decoder 当前层的隐状态为 $D \in \mathbb{R}^{m \times d}$，Encoder 输出为 $Z \in \mathbb{R}^{n \times d}$：

```math
Q = D\,W_Q, \qquad K = Z\,W_K, \qquad V = Z\,W_V
```

```math
\mathrm{CrossAttn}(D, Z) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
```

**维度核对**（严格，必须在心里过一遍）：$Q \in \mathbb{R}^{m \times d_k}$，$K, V \in \mathbb{R}^{n \times d_k}$，打分矩阵 $QK^\top \in \mathbb{R}^{m \times n}$——**目标序列的每个位置，对源序列的每个位置都有一份权重**；输出 $\in \mathbb{R}^{m \times d}$。注意这个分工是结构定义，严格成立：

- $Q$ 来自 Decoder（"我当前想知道什么"）；
- $K$、$V$ 来自 Encoder（"源语言里有什么可查"）。

Decoder block 的完整数据流（Pre-LN 写法，$x$ 为本层输入）：

```math
\begin{aligned}
h_1 &= x + \mathrm{MaskedAttn}(\mathrm{LN}(x)) \\
h_2 &= h_1 + \mathrm{CrossAttn}(\mathrm{LN}(h_1),\, Z) \\
y &= h_2 + \mathrm{FFN}(\mathrm{LN}(h_2))
\end{aligned}
```

三个子层、两次残差，维度全程守恒为 $d$。

**数值示例。** 设 $d_k = 2$。Decoder 正在生成第 1 个目标词，当前只有一个 query 向量 $q = [\,1,\; 0\,]$；Encoder 输出了 2 个位置：$k_1 = [\,1,\; 0\,]$（源词"我"），$k_2 = [\,0,\; 1\,]$（源词"猫"），$v_1 = [\,2,\; 3\,]$，$v_2 = [\,4,\; 1\,]$。

打分：$q \cdot k_1 = 1$，$q \cdot k_2 = 0$；除以 $\sqrt{2} \approx 1.4142$ 得 $[\,0.7071,\; 0\,]$。

softmax：$e^{0.7071} \approx 2.0281$，$e^0 = 1$，和为 $3.0281$，权重 $\approx [\,0.6698,\; 0.3302\,]$。

输出 $= 0.6698\,v_1 + 0.3302\,v_2 = [\,0.6698 \times 2 + 0.3302 \times 4,\;\; 0.6698 \times 3 + 0.3302 \times 1\,] \approx [\,2.6604,\; 2.3396\,]$。

解读（直觉）：生成这个词时，模型把约 67% 的注意力给了源词"我"（因为 $k_1$ 与 $q$ 更对齐），输出是两个源位置 value 的加权混合。"权重是概率分布（和为 1）"是 softmax 定义的严格结论；"67% 注意力意味着模型在'想'这个词"是直觉解读，需实验验证。

## 4. 组装：完整的 Encoder-Decoder 数据流

把翻译"我爱猫"→"I love cats"完整走一遍：

1. 源句分词 + 词嵌入 + 位置编码 → $X \in \mathbb{R}^{3 \times d}$；
2. Encoder（$N$ 层 block，无 mask）→ $Z \in \mathbb{R}^{3 \times d}$，每个 $z_i$ 融合整句；
3. Decoder 输入起始符 `<bos>`：因果自注意力处理已生成部分，对 $Z$ 做交叉注意力查源句 → 输出层预测 "I"；
4. 把 "I" 拼回 Decoder 输入，重复第 3 步 → "love" → "cats" → 预测出 `<eos>` 停止。

~~~mermaid
flowchart TB
    SRC["源序列 X ∈ ℝ^{n×d}"] --> ENC["Encoder × N 层<br/>无 mask 自注意力"]
    ENC --> Z["上下文表示 Z ∈ ℝ^{n×d}"]
    TGT["目标前缀 y_{<t}"] --> DEC["Decoder × N 层<br/>因果自注意力 + 交叉注意力"]
    Z --> DEC
    DEC --> LOGITS["词表 logits ∈ ℝ^{|V|}"]
    LOGITS --> NEXT["采样 y_t，拼回输入"]
    NEXT --> TGT
~~~

**训练并行 vs 生成串行**（严格论证，第 04 篇结论在这里的应用）：

- **训练时**目标序列 $Y$ 完整已知（teacher forcing 机制，第 10 篇会细讲）。因果 mask 保证位置 $t$ 的损失只依赖 $y_{\le t}$——计算图中不存在"未来 → 过去"的依赖边，所以 $m$ 个位置可以**一次前向同时算出**，充分利用 GPU 并行；
- **生成时** $y_t$ 本身是待求的未知量：不算出 $y_t$ 就无法构造位置 $t+1$ 的输入，依赖链是串死的——**必须 $m$ 次前向**，天然串行。

**参数量估算**（严格计数，取原论文 base 配置 $d = 512$、$N = 6$、词表 $|V| = 32000$，只计权重矩阵）：

| 部分 | 每层参数量 | $N = 6$ 层合计 |
|---|---|---|
| Encoder（自注意力 $4d^2$ + FFN $8d^2$） | $12d^2 \approx 3.15\text{M}$ | $\approx 18.9\text{M}$ |
| Decoder（多一个交叉注意力 $4d^2$） | $16d^2 \approx 4.19\text{M}$ | $\approx 25.2\text{M}$ |
| 词嵌入 $\|V\| \times d$ | — | $\approx 16.4\text{M}$ |

合计约 60M，与原论文报告的 base 模型 65M 量级一致（差值来自偏置、LayerNorm 参数与输出层是否共享嵌入；"65M"是论文报告的实验配置事实）。从计数可以直接读出：Decoder 一层恰为 Encoder 一层的 $16/12 = 4/3$ 倍——多的正是交叉注意力。

## 5. 变体：只用一半也成立——Encoder-only 与 Decoder-only

**严格事实**：Encoder 和 Decoder 可以独立使用，对应不同性质的任务：

| 结构 | 自注意力 mask | 擅长的任务 | 代表模型 |
|---|---|---|---|
| Encoder-only | 无（双向） | 阅读理解、分类、句向量 | BERT |
| Decoder-only | 因果（单向） | 文本生成、对话 | GPT |
| Encoder-Decoder | Encoder 无、Decoder 因果 + 交叉 | 翻译、摘要 | T5、原版 Transformer |

**直觉**：理解类任务希望每个词看到整句（"它"指代谁必须看完全句），生成类任务必须单向（不能偷看未来）。现代大语言模型多为 Decoder-only——"Decoder-only 足以做好生成"是**实验验证**的结论（直觉上也说得通：生成任务的输入输出是同一序列，不需要一套独立的"阅读器"）。

**条件辨析**："哪种结构在某任务上更好"是实验结论，不是结构定理。比如 BERT 在理解任务上的优势是 Devlin et al. (2019) 报告的实验结果；GPT 系列在生成上的成功同样是实验事实。结构只决定"能看到什么信息"，不决定"效果一定好"。

## 6. 常见误区

**误区 1：交叉注意力的 Q、K、V 都来自 Decoder，只是名字不同。**

纠正：不是。$Q$ 来自 Decoder（当前生成状态，"我想查什么"），$K$、$V$ 来自 Encoder 输出（源语言表示，"字典里有什么"）。打分矩阵是 $m \times n$（目标 × 源）的长方形，不是方阵——如果算出来是方阵，说明理解错了。

**误区 2：Decoder 生成时每一步都要把 Encoder 重新算一遍。**

纠正：不需要。Encoder 输出 $Z$ 在生成开始前算一次，之后每一步的交叉注意力都复用同一份 $Z$（$K$、$V$ 不变，只有 $Q$ 随新生成的词变化）。重复计算 Encoder 是纯浪费——第 08 篇的 KV Cache 会把这种"复用"思想推到极致。

**误区 3：Encoder-Decoder 一定比 Decoder-only 好，因为"零件更多"。**

纠正：零件多不等于效果好。结构选择取决于任务形态：翻译、摘要这种"输入输出是两种序列"的任务适合 Encoder-Decoder；纯生成任务上 Decoder-only 经实验验证足够好。优劣是实验结论，不是参数量竞赛。

**误区 4：训练时 Decoder 像生成时那样一次只看一个词、逐词往前跑。**

纠正：不是。训练时目标序列完整已知（teacher forcing），因果 mask 让 $m$ 个位置一次前向同时计算，数学上等价于逐词生成，但吃满了 GPU 并行。这是第 04 篇的核心结论，这里是它的直接应用。

## 7. 自测题

**Q1.** 交叉注意力中 $Q \in \mathbb{R}^{m \times d_k}$，$K, V \in \mathbb{R}^{n \times d_k}$。写出 $Q$、$K$、$V$ 分别由什么经过什么投影得到，打分矩阵的形状是什么？

<details>
<summary>答案</summary>

$Q = D\,W_Q$（$D$ 为 Decoder 当前层隐状态）；$K = Z\,W_K$，$V = Z\,W_V$（$Z$ 为 Encoder 输出）。打分矩阵 $QK^\top \in \mathbb{R}^{m \times n}$：目标序列每个位置对源序列每个位置的权重（成立条件：$W_Q$ 与 $W_K$ 的输出维度同为 $d_k$）。
</details>

**Q2.** 用第 3 节数值示例的数据（$q = [1, 0]$，$k_1 = [1, 0]$，$k_2 = [0, 1]$）：若去掉 $\sqrt{d_k}$ 缩放，权重会变成多少？这说明了缩放的什么作用？

<details>
<summary>答案</summary>

未缩放打分为 $[\,1,\; 0\,]$，softmax 得 $[\,e/(e+1),\; 1/(e+1)\,] \approx [\,0.7311,\; 0.2689\,]$，比缩放后的 $[\,0.6698,\; 0.3302\,]$ 更尖锐。缩放的作用（第 02 篇结论）：防止 $d_k$ 较大时打分方差过大导致 softmax 饱和、梯度消失。
</details>

**Q3.** 用一句话各自给出严格理由：为什么 Transformer 训练可以并行、生成必须串行？

<details>
<summary>答案</summary>

训练：目标序列完整已知，因果 mask 使位置 $t$ 的计算只依赖 $y_{\le t}$，计算图无"未来 → 过去"的依赖边，$m$ 个位置可一次前向并行计算。生成：$y_t$ 本身是待求未知量，不算出它就无法构造 $t+1$ 的输入，依赖链天然串行。
</details>

**Q4.** 取 base 配置（$d = 512$，$N = 6$），只计注意力与 FFN 的权重矩阵：Decoder 一层的参数量为什么是 Encoder 一层的 $4/3$ 倍？

<details>
<summary>答案</summary>

Encoder 每层：自注意力 $4d^2$ + FFN $8d^2$ = $12d^2$；Decoder 每层多一个交叉注意力 $4d^2$，共 $16d^2$。$16/12 = 4/3$（严格计数，忽略偏置与 LayerNorm 参数时成立）。
</details>

---

**下一篇预告**：第 08 篇《KV Cache：推理时为什么能省计算》——生成时每一步都在重复算什么？把 K、V 缓存起来到底能省掉多少计算量？
