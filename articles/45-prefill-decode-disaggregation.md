# 45｜PD 分离：prefill 和 decode 为什么要分到不同机器上

> **阅读目标**：理解为什么 prefill 和 decode 混跑在一张卡上会互相伤害（干扰）；能从 DistServe 的实测数字出发推导"分到两个 GPU 池"为什么能把每卡 goodput 提高约 2 倍，并算出 Llama-3.1-8B 下一个 8192 token 提示词的 KV 缓存跨卡传输要花多少时间、以及 prefill 池和 decode 池的配比该怎么定。
>
> **前置知识**：第 02 篇的缩放点积注意力（Q、K、V 投影与打分）；第 04 篇的因果 mask 与自回归生成（prefill 一次性算全部 token、decode 一次算一个 token）；第 08 篇的 KV Cache（prefill 是算力 bound、decode 是显存带宽 bound）；第 37 篇的 PagedAttention（KV 缓存按块分配）；第 38 篇的连续批处理（请求随到随跑、每步重新组 batch）；第 44 篇的 Chunked Prefill（分块 prefill 只是把干扰摊薄、不是消除）。

## 1. 直觉：同住一张卡的两个房客，作息完全相反

第 44 篇结尾留了一个问题：分块 prefill 把一次 330 毫秒的卡顿摊薄成每步 22 毫秒，但每步的 22 毫秒仍然是干扰。想把干扰清零，就得把两个阶段彻底分开——**PD 分离（Prefill-Decode Disaggregation）**：prefill 池只算提示词，decode 池只吐 token，KV 缓存随请求在机器之间传输。

为什么两个阶段值得住两套房子？因为它们的作息完全相反。第 08 篇和第 44 篇已经算过这笔账：

- **Prefill 是爆发户。** 一次处理全部 $T$ 个 token，主导成本是 $2PT$ 的矩阵乘法 FLOPs，算术强度高，张量核心跑满，瓶颈是算力。它来得快、去得快，一段 8192 token 的提示词在 8B 模型上约 0.33 秒算完，然后就消失了。
- **Decode 是长住户。** 每步每个请求只算 1 个新 token，主导成本是把全部权重和该请求的全部 KV 缓存从 HBM 读一遍，瓶颈是显存带宽，算力大量闲置。它一次只吐一个 token，却要住上几十甚至几百步——一个 200 token 的回答，就是 200 步。

把它们塞进同一张卡、同一个 batch，等于让一个通宵赶工的装修队和一个要安静睡觉的房客合租：装修队一开工（大 prefill 进 batch），房客的睡眠（decode 的 token 间隔）就被打断；房客要睡得安稳（ITL SLO），装修队就得小声干活（chunked prefill 限预算），工期就被拖长。**第 44 篇的分块 prefill 是"协商作息表"，PD 分离是"各住各的房子"**——代价是 KV 缓存要在两套房子之间搬一次家。

这个想法在 2023—2024 年由两篇论文先后落地：Splitwise（Patel 等，arXiv:2311.18677，2023 年）把 prefill 和 decode 分到不同的机器池，并利用两阶段对硬件需求的不同来省钱；DistServe（Zhong 等，arXiv:2401.09670，OSDI 2024）把分离做成一套完整的 serving 系统，用模拟器按 SLO 自动算出每池该配多少卡、每阶段该用什么并行策略。两篇论文的结论方向一致：**分离不改变任何一次计算，只改变"谁和谁共享一张卡"**，而这个改变本身就能让每张卡多服务一倍以上的请求。

> 结论性质：下面"分离不改变任何数值结果"是**严格推导**（第 3 节末尾）；"goodput 提高约 2 倍"是**论文实测**（DistServe 图 1 的 13B/A100 设定，见第 3 节），换模型、换负载数字会变；"KV 传输耗时"是**严格推导的算术**加上**经验性**的硬件假设（H100 有效算力按 400 TFLOPS、HBM 带宽 3.35 TB/s、NVLink 约 900 GB/s，见第 5 节）；vLLM 等引擎的 PD 模式是**工程事实**，以官方文档为准，版本升级可能变化。

## 2. 两个阶段想要的东西完全不同

分离之前，先把"不同"量化到能指导配卡的程度。两个阶段在四个维度上全都相反：

| 维度 | Prefill | Decode |
|---|---|---|
| 瓶颈资源 | 算力（FLOPs） | 显存带宽 + KV 容量 |
| batch 的收益 | 一个请求内部已有 $T$ 个 token 并行，加 batch 收益递减 | 每步只算 1 token/请求，batch 越大、权重读取摊得越薄 |
| 偏好的并行策略 | 张量并行（TP）：把一次前向切开、缩短 TTFT | 大 batch + 流水线并行（PP）：多请求分摊权重读取 |
| 偏好的硬件 | 最新、最强算力的卡（H100） | 算力弱一点也没关系，带宽和容量够用就行（A100、甚至降频的 H100） |

**第四行是 Splitwise 最关键的洞察。** 硬件代际升级是不均匀的：Splitwise 论文的表 I 量过，从 A100 到 H100，算力涨了 3.43 倍、功耗涨了 1.75 倍，显存带宽只涨了 1.6 倍、容量根本没涨。也就是说，**每一代新卡，算力都比带宽跑得快**——prefill 那个算力 bound 的阶段，每代都能吃到红利；decode 那个带宽 bound 的阶段，每代只分到一点点。买一台最新最强的卡同时干两件事，等于用买算力的钱去迁就一个不吃算力的阶段。

分离之后，这个错配可以反过来利用：prefill 池用 H100（算力贵但值得），decode 池用 A100 或者把 H100 功耗上限调低一半（decode 本来就用不满算力，降频几乎不损失速度，却省电省钱）。Splitwise 报告的结果是：**同等成本与功耗下吞吐量 2.35 倍，或者吞吐量提高 1.4 倍的同时成本降低 20%**。注意这两个数字的前提——"同等成本"和"吞吐提高但成本降低"是两组不同的配置，不是可以同时拿到的两份好处。把它们读成"又快又便宜"就误解了论文。

还有一个维度容易被忽略：**两个阶段的 SLO 也不一样**。Prefill 对应的是 TTFT（首 token 延迟），用户能忍受几百毫秒；decode 对应的是 TPOT/ITL（每 token 间隔），用户对卡顿极其敏感，几十毫秒的抖动就能感觉到。混跑时，一个 SLO 的压力会传染给另一个：第 44 篇的例子里，8192 token 的 prefill 让 32 个人的 ITL 从 10 毫秒 spike 到 330 毫秒——TTFT 的需求（快点算完长提示词）直接打穿了 ITL 的 SLO。分离之后，两个 SLO 各自由自己的池子负责，互不传染。

## 3. 严格论证：为什么分离后每卡能多服务一倍请求

先定义衡量标准。DistServe 引入的概念叫 **goodput（有效吞吐）**：在 TTFT 和 TPOT 两个 SLO 都满足的前提下（比如 90% 的请求达标），系统每秒能服务的最大请求数，再除以 GPU 数，就是**每卡 goodput**。它和裸吞吐的区别在于：裸吞吐只数"算了多少"，goodput 只数"按时算完多少"——一个为了冲吞吐把 ITL 拖到 500 毫秒的系统，goodput 是零。

DistServe 论文图 1 给了一个干净到可以手算的例子：13B 模型、单张 A100、输入 512 token、输出 64 token，SLO 要求 90% 的请求同时满足 TTFT 和 TPOT 目标。

- **混跑**：一张卡既做 prefill 又做 decode，每卡 goodput 约 **1.6 请求/秒**。瓶颈是两者中更严格的那个 SLO——prefill 挤占 decode 的步，TPOT 先顶不住。
- **只做 prefill**：一张卡只算 prefill，每卡 goodput **5.6 请求/秒**。
- **只做 decode**：一张卡只做 decode，每卡 goodput **10 请求/秒**。
- **分离**：2 张卡做 prefill、1 张卡做 decode，系统整体 10 请求/秒，即每卡 **3.3 请求/秒**，是混跑的 **2.1 倍**。

为什么是 2.1 倍而不是 5.6 倍？因为系统整体的 goodput 由两个池子中**更慢的那个**决定：2 张 prefill 卡提供 $2 \times 5.6 = 11.2$ 请求/秒的能力，1 张 decode 卡提供 10 请求/秒，整体被 decode 卡住、约 10 请求/秒。分母是 3 张卡，所以每卡 3.3。这个例子同时给出了**配比的算法**：设 prefill 单卡 goodput 为 $g_p$、decode 单卡 goodput 为 $g_d$，要服务 $R$ 请求/秒，需要 $R / g_p$ 张 prefill 卡和 $R / g_d$ 张 decode 卡，池子配比就是 $g_d : g_p$——这里是 $10 : 5.6 \approx 1.8 : 1$，约等于 2:1。配比不是拍脑袋定的，是两个单卡 goodput 的比值倒数。

更一般地，混跑的每卡 goodput 之所以低，是因为**两个 SLO 在同一张卡上互相掣肘**：prefill 需要大块连续算力、decode 需要稳定短步长，任何一个 SLO 收紧，整张卡的 goodput 就被它拖下来。分离把约束解耦：prefill 卡只对 TTFT 负责、decode 卡只对 TPOT 负责，每张卡的 goodput 只受自己的 SLO 约束。论文摘要报告，在多组模型与 SLO 组合下，分离最高可服务 **4.48 倍**的请求量，或满足 **10.2 倍**更严格的 SLO——这两个是不同配置下的最高值，不是图 1 那个例子的 2.1 倍，别把最高值当成典型值。

还有一个经常被漏掉的前提：**分离不改变任何数值结果**。Decode 池收到的 KV 缓存，就是 prefill 池算出来的同一份 K、V（第 5 节会算它有多大）；decode 的每一步注意力，读到的 K、V 与混跑时完全相同。RoPE 的旋转角只依赖绝对位置（第 05 篇），KV 在哪张卡上算出来都不改变它的值。所以 PD 分离和第 44 篇的 chunked prefill 一样，是**调度与放置算法，不是近似算法**——输出分布严格不变（浮点传输不改变数值，跨卡求和顺序的差异同样只有最后一位量级的噪声）。

## 4. 搬家费：KV 缓存怎么跨卡

分离的全部代价，集中在一件事上：**每个请求的全部 KV 缓存要从 prefill 池搬到 decode 池**。搬家费由三笔账组成：搬多少、走什么路、什么时候搬。

**搬多少。** KV 缓存的大小只取决于模型和上下文长度，与分离无关。第 43 篇算过，Llama-3.1-8B 每 token 的 KV 是 128 KiB（32 层 × 8 个 KV 头 × 128 维 × K/V 两份 × fp16 两字节）。所以搬家费的公式是：

```math
\text{KV 字节数} = T \times 2 \times L \times H_{kv} \times d_h \times \text{bytes}
```

$T$ 是提示词长度。8192 token 的提示词要搬 $8192 \times 128 \text{ KiB} = 1$ GiB（约 1.07 GB）；70B 模型（每 token 320 KiB）同样长度要搬约 2.5 GiB。注意这是**一次性**的费用：搬完之后，decode 的每一步只在 decode 池本地追加 1 个 token 的 KV，不再跨卡。

**走什么路。** 同一节点内的两张卡走 NVLink，跨节点走 InfiniBand 或以太网，带宽差两个数量级——搬家费的天壤之别全在这里。第 5 节的表会给出具体数字。DistServe 的放置算法（placement）核心就是这一条：**尽量把配对的 prefill 实例和 decode 实例放在同一个节点内**，让传输走 NVLink；跨节点的配对只在节点内放不下时才考虑。Splitwise 更进一步：它把 KV **逐层传输**——prefill 每算完一层的 K、V 就立刻发往 decode 池，而不是等全部 32 层算完再一次性搬。这样传输时间被重叠进 prefill 的计算时间里，首 token 延迟里几乎看不到传输的影子。

**什么时候搬。** 传输发生在 prefill 完成、decode 开始之前，所以它直接加进 TTFT 里：

```math
\text{TTFT} = T_{\text{排队}} + T_{\text{prefill 计算}} + T_{\text{KV 传输}}
```

第 5 节会看到，NVLink 下 $T_{\text{KV 传输}}$ 只有约 1 毫秒，相对几百毫秒的 prefill 计算可以忽略；但如果走的是 100 GbE，传输要约 86 毫秒，已经占到 TTFT 的两成——这时分离的账就算不过来了。**PD 分离成立的前提，第一条就是互联带宽足够快。**

工程上，主流引擎已经把这套流程产品化：vLLM 提供 disaggregated prefilling 模式（KV 传输走 NIXL 等连接器），SGLang、TensorRT-LLM 也有对应的 PD 分离部署形态；Mooncake（arXiv:2407.00079，2024 年）则把思路再推一步：KV 缓存不只在 prefill 与 decode 之间搬，还汇入一个全集群共享的 CPU 内存/SSD 池，变成跨请求的共享资产——那是下一篇的主题。

```mermaid
flowchart LR
    A[请求到达] --> B[prefill 池<br/>H100, 算力优先]
    B --> C[逐层传输 KV<br/>NVLink / InfiniBand]
    C --> D[decode 池<br/>A100 或降频 H100, 带宽优先]
    D --> E[逐 token 生成]
    E -- "上下文变长,<br/>KV 在本地追加" --> E
```

## 5. 数值示例：同一个 8192 token 提示词，搬家要花多久

拿第 43、44 篇同一个模型算账：Llama-3.1-8B（32 层、8 个 KV 头、头维 128、fp16），每 token KV 128 KiB。场景：一个 8192 token 的长提示词，KV 共 1 GiB（约 1.07 GB）。硬件按 H100 计：fp16 有效算力约 400 TFLOPS、HBM 带宽 3.35 TB/s。

**先算 prefill 本身。** 第 44 篇算过：8192 token 的 prefill FLOPs 约 $2 \times 8\times10^9 \times 8192 \approx 1.3\times10^{14}$，约 0.33 秒。这是 TTFT 里的大头，传输时间要和它比大小。

**再算四条路的搬家时间**（理论带宽下限，实际带宽受协议开销与竞争影响，只会更慢）：

| 互联 | 有效带宽 | 传输 1.07 GB | 占 TTFT（约 330 ms）的比例 |
|---|---|---|---|
| NVLink（H100 节点内） | 约 900 GB/s | 约 1.2 ms | 约 0.4% |
| InfiniBand 400 Gb/s | 约 50 GB/s | 约 21 ms | 约 6% |
| PCIe Gen4 x16 | 约 25 GB/s | 约 43 ms | 约 13% |
| 100 GbE 以太网 | 约 12.5 GB/s | 约 86 ms | 约 26% |

验算一行：InfiniBand 400 Gb/s $= 400 / 8 = 50$ GB/s，$1.07 / 50 \approx 0.021$ 秒 $= 21$ 毫秒。NVLink 行：$1.07 / 900 \approx 0.0012$ 秒。

这张表是整篇最值得记住的结论：**KV 传输本身不是问题，互联才是问题**。节点内 NVLink 下，搬家费只有 1 毫秒级，相对 330 毫秒的 prefill 计算完全可以忽略——DistServe 论文也报告过，95% 以上的请求传输延迟在 30 毫秒以内、占总延迟不到 0.1%（OPT-175B 的设定）。但一旦跨节点走到普通以太网，86 毫秒的传输已经吃掉 TTFT 的四分之一，分离的收益就被搬家费吃掉大半。这就是为什么 DistServe 的放置算法把"同节点配对"当成硬约束、而不是优化项。

**再算配比。** 沿用第 3 节的设定（13B、512 输入、64 输出、A100）：prefill 单卡 goodput 5.6、decode 单卡 10，配比 2:1。换一个负载形状，配比跟着变：

- **RAG/长文档总结**（输入 8192、输出 256）：prefill 变重，单卡 prefill goodput 下降，配比向 prefill 倾斜，比如 3:1 甚至 4:1。
- **多轮聊天**（输入 1024、输出 512）：decode 变重，decode 池要更多卡，配比向 decode 倾斜，比如 1:1 甚至 1:2。
- **前缀缓存命中率高时**（第 43 篇）：大量请求的 prefill 被缓存跳过，prefill 池的压力骤降，配比同样向 decode 倾斜。

所以 PD 分离的运维现实是：**配比不是一次定死的常数，是跟着负载形状走的变量**。DistServe 用模拟器按历史 trace 定期重算配比；负载形状突变（比如晚高峰从短问答切成长文档）时，配比滞后就会让某一池先顶不住——这是第 6 节的代价之一。

## 6. 代价：分离不是免费的午餐

**第一笔：搬家费本身。** 第 5 节的表已经说明：节点内可忽略，跨节点以太网下占 TTFT 的四分之一。更糟的是长上下文：70B 模型、32768 token 的提示词，KV 约 10 GiB，走 InfiniBand 也要约 200 毫秒——这时即使传输与计算重叠（逐层传），最后一层的 KV 仍要在 decode 开始前到位，TTFT 里至少要留出"最后一层传输 + 调度"的零头。

**第二笔：权重与调度的双份开销。** 两个池子各存一份完整权重（8B fp16 每卡 16 GB），集群总的权重显存占用不变（每卡本来就要存一份），但**调度器变成两个**：prefill 池的调度器管 TTFT、decode 池的调度器管 TPOT，中间还要一个桥接队列（bridge queue）做请求的交接。请求的状态从"在一个引擎里"变成"跨两个引擎"，失败处理（prefill 池挂了、传输中途断了、decode 池满了）都要重新设计。单机部署时这些都不存在。

**第三笔：配比失衡。** 第 5 节末尾讲过，配比跟着负载形状走。负载形状是按小时变的：白天短问答多、晚上长文档多；一个热点事件能让输入长度分布在一小时内翻几倍。配比调慢了，某一池先排队——prefill 池排队，TTFT 涨；decode 池排队，请求在桥接队列里等着、首 token 出不来。DistServe 靠模拟器 + 历史 trace 提前算配比，但模拟器假设"负载形状在几小时内稳定"，负载突变时这个假设就失效。这是分离相对混跑最实在的运维代价：**混跑的配比是自动的（同一张卡、同一个 batch），分离的配比是人工的（要算、要调、要跟上负载）**。

**第四笔：离线批量任务几乎没有收益。** DistServe 论文自己指出：干扰本质上是延迟问题——prefill 拖慢 decode 的 ITL，才需要分离。如果任务没有延迟 SLO（离线批量推理、评测集跑分），混跑的大 batch 本来就是最优的：prefill 的算力闲置被 decode 的 batch 填满，分离反而多付一份传输费。**PD 分离是给在线服务的架构，不是给离线任务的。**

四笔账加起来，PD 分离的定位就清楚了：**它是在线推理在 SLO 约束下的最优解，不是普适更快的推理**。单机小流量时，chunked prefill（第 44 篇）已经够用；跨节点、大流量、TTFT 与 TPOT 都有硬 SLO 时，才值得付搬家费和配比管理的代价。想再进一步——让 KV 缓存不只搬一次、而是变成全集群共享、跨请求复用的资产——那是 Mooncake 的 KV 缓存池化，下一篇细讲。

## 7. 常见误区

**误区 1："PD 分离让推理变快了。"**

错。总 FLOPs 不变，单个请求的 TTFT 反而略变长（多了 KV 传输的那几毫秒到几十毫秒）。它变好的是**每卡 goodput**——在 SLO 约束下每张卡能多服务约一倍的请求。把"单位成本能服务的请求变多"说成"推理变快"，方向就反了。

**误区 2："KV 缓存 1 GB，跨卡传输肯定是瓶颈。"**

错。1 GB 听起来大，但 NVLink 下只要约 1 毫秒——相对 330 毫秒的 prefill 计算只有 0.4%。瓶颈不是传输量，是**互联带宽**：同一条 1 GB 的缓存，走 NVLink 是 1 毫秒、走 100 GbE 是 86 毫秒。问"传输是不是瓶颈"之前，先问"走什么路"。

**误区 3："分离后 prefill 和 decode 可以用不同的模型。"**

错。两个池子跑的是**同一份权重**：decode 池收到的 KV 是 prefill 池用同一模型算出来的，换模型 KV 就对不上了。分离改变的是放置、并行策略和硬件选型，不是模型本身。两个池子的权重版本还必须严格同步——prefill 池先升级、decode 池后升级，中间混跑的请求 KV 就会错位。

**误区 4："有了 PD 分离，chunked prefill 就没用了。"**

错。两者是不同层面的答案：PD 分离解决**跨池**的干扰（prefill 不再进 decode 的 batch），chunked prefill 解决**池内**的干扰（prefill 池内部，多个长提示词之间、以及 prefill 池自己的步长控制）。Prefill 池内部同样有"一个超长提示词拖住后面所有 prefill"的排队问题，chunked prefill 在池内照样工作。两者是组合关系，不是替代关系。

**误区 5："离线批量推理也该上 PD 分离。"**

错。没有延迟 SLO 就没有干扰问题——第 44 篇的卡顿（ITL spike）之所以是问题，是因为用户在实时等 token；离线任务没有人在等，混跑的大 batch 反而是算力利用率最高的形态。分离的收益只在"有 SLO 要满足"的前提下成立。

## 8. 自测题

**Q1.** DistServe 图 1 的设定：13B 模型、A100、输入 512、输出 64。混跑每卡 goodput 约 1.6 请求/秒；只做 prefill 约 5.6、只做 decode 约 10。若配 2 张 prefill 卡 + 1 张 decode 卡，系统整体 goodput 是多少？每卡 goodput 是多少？是混跑的几倍？瓶颈在哪一池？

<details>
<summary>答案</summary>

Prefill 池能力 $2 \times 5.6 = 11.2$ 请求/秒，decode 池能力 $1 \times 10 = 10$ 请求/秒，系统整体被较慢的 decode 池卡住、约 10 请求/秒。共 3 张卡，每卡 goodput $10 / 3 \approx 3.3$ 请求/秒，是混跑 1.6 的约 2.1 倍。瓶颈在 decode 池——再加 prefill 卡也无济于事，要加 decode 卡（或等负载形状变化后重算配比）。

</details>

**Q2.** Llama-3.1-8B，8192 token 的提示词，KV 缓存共约 1.07 GB。分别算它走 NVLink（约 900 GB/s）和 100 GbE（约 12.5 GB/s）要花多少毫秒？各占约 330 毫秒 TTFT 的百分之几？由此说明 PD 分离对互联有什么要求。

<details>
<summary>答案</summary>

NVLink：$1.07 / 900 \approx 0.0012$ 秒 $\approx 1.2$ 毫秒，约占 TTFT 的 0.4%。100 GbE：$1.07 / 12.5 \approx 0.086$ 秒 $\approx 86$ 毫秒，约占 TTFT 的 26%。结论：KV 传输本身不是瓶颈，互联带宽才是——分离成立的前提是 prefill 池与 decode 池之间有高速互联（节点内 NVLink 或 InfiniBand），走普通以太网时搬家费会吃掉分离的大部分收益。

</details>

**Q3.** 为什么 PD 分离之后，单个请求的 TTFT 反而略变长，但系统的每卡 goodput 却提高了约 2 倍？用第 3 节的约束解耦来解释。

<details>
<summary>答案</summary>

TTFT 变长是因为多了 $T_{\text{KV 传输}}$ 这一项（NVLink 下约 1 毫秒，以太网下几十毫秒）。Goodput 提高是因为约束解耦：混跑时一张卡要同时满足 TTFT 和 TPOT 两个 SLO，prefill 挤占 decode 的步，TPOT 先顶不住，整卡 goodput 被更严格的那个 SLO 拖到 1.6；分离后 prefill 卡只对 TTFT 负责、decode 卡只对 TPOT 负责，每张卡的 goodput 只受自己的 SLO 约束，合计达到每卡 3.3。TTFT 是单个请求的账，goodput 是整卡的账——两者变好的方向不一样。

</details>

**Q4.** 一个在线服务白天主要是短问答（输入 512、输出 64），晚上主要是长文档总结（输入 8192、输出 256）。如果 prefill:decode 池的配比白天是 2:1，晚上该往哪个方向调？为什么？前缀缓存命中率突然升高时，配比又该往哪个方向调？

<details>
<summary>答案</summary>

晚上往 prefill 方向调（比如 3:1 或 4:1）：输入变长 16 倍，prefill 单卡 goodput 下降，需要更多 prefill 卡才能跟上；输出只变长 4 倍，decode 池压力增长相对较小。反过来，前缀缓存命中率升高时，往 decode 方向调：大量请求的 prefill 被缓存跳过（第 43 篇），prefill 池压力骤降，瓶颈移到 decode 池，需要更多 decode 卡。配比不是常数，是跟着输入/输出长度分布和缓存命中率走的变量。

</details>

---

下一篇预告：第 46 篇将讲 **Mooncake 与 KV 缓存池化：KV 缓存为什么不该只属于一个请求**——PD 分离让 KV 在 prefill 与 decode 之间搬一次家，Mooncake 则把它汇入全集群共享的 CPU 内存/SSD 池：跨请求的相同前缀只存一份、冷数据沉到 SSD、热数据留在显存，KV 缓存从"请求的私有财产"变成"集群的共享资产"。

---

**参考文献**

- Patel, P. 等，Splitwise: Efficient Generative LLM Inference Using Phase Splitting，arXiv:2311.18677，2023 年。
- Zhong, Y. 等，DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving，arXiv:2401.09670，OSDI 2024。
- Agrawal, A. 等，Sarathi-Serve: Efficiently Serving Large Language Models by Efficiently Scheduling Prefill and Decode，arXiv:2308.16369，OSDI 2024（chunked prefill 的来源，第 44 篇已引用）。
- Qin, R. 等，Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving，arXiv:2407.00079，2024 年。
