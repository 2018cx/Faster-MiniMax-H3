# Faster-MiniMax-H3

[English](README.md) | [简体中文](README_zh.md)

**在 8×H20 GPU 上将 MiniMax-H3 LoRA 微调 optimizer step 加速 1.86×，由 AI agents 完成优化。**

| 实现 | 每个 optimizer step 时间（秒）↓ | 加速比 ↑ |
|---|---:|---:|
| Reference | 117.318886 | 1.00× |
| **Faster-MiniMax-H3** | **63.176236** | **1.8570×** |

正式结果来自 BF16 FL2VA LoRA/ZeRO-3 的 reference/candidate/reference 检查：每个 rank batch 为 1，
每个样本约 33,750 token。独立的三 seed × reference/optimized 配对验证共六个 arm，每个 arm 恰好
运行 50 个 optimizer step，8 项预先声明的比较全部通过。这是短程缓存工作负载结果，不代表完整收敛
或生成质量结论。

Faster-MiniMax-H3 是 [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) 面向 H20
训练的 agent 优化版本，包含基于官方 Diffusers 集成的监督式 rectified-flow trainer、latent 缓存
预处理，以及 [FIXES.md](FIXES.md) 中记录的 LoRA × ZeRO-3 正确性修复。

## 版本选择与验证

最终发布的不是原始 Round 19，而是 **Round 50 首次通过正式速度 check 的修复候选**；由于后续
修复候选都没有达到 1.8691× 的提前停止目标，它在 **Round 60 收尾时被正式选中**。8×H20 上的
reference/candidate/reference 正式测量为 **63.176236 秒/step 对 117.318886 秒/step，即
1.8570097438×**。公开运行时代码 commit 是
[`7d7be36`](https://github.com/2018cx/Faster-MiniMax-H3/commit/7d7be367a2611193b354570a909c8fcc85216394)，
与原始审计 ID `5c47dee7974a8b69990f05ab4e32cf298bf4ee7e` 的代码树完全一致，仅发布署名元数据不同。

Round 19 没有发布，是因为它虽然以 1.869055× 通过两步速度检查，随后却在三 seed、每 arm 50 step
验证中出现间歇性的有限梯度尖峰。第 25 step 的 optimized 平均 pre-clip 梯度 L2 为 895,829.90，
reference 为 10.68，绝对标准化均值差达到 142,950.33；seed 424243 在第 18 step 也出现
11,248,900.41 对 5.95 的尖峰。梯度裁剪让更新仍保持有限，但轨迹一致性门槛明确失败。

定向诊断随后发现了 checkpoint replay 不变量缺陷：8 个 rank 中有 5 个在 Q/K RMSNorm 输入、权重、
epsilon 和 stride 完全相同时，因为 inference 与 replay 选择了不同 XBLOCK，编译输出仍不一致。
修复方案让两种 checkpoint 模式共享同一个 detached compiled Q/K forward，并保留既有的编译一阶
导数路径。这个结果证明了一个具体缺陷，但不声称它一定是历史尖峰的唯一原因。修复后重新执行了
三 seed × reference/optimized 共六个 arm、每 arm 恰好 50 step 的验证，8 项预先声明的比较全部通过，
最大绝对标准化均值差为 1.4698 ≤ 2.0。详见[修复审计](docs/repair-results.md)、
[原始失败验证](docs/validation-results.md)和[修复后验证](docs/repaired-validation-results.md)。

原始 Round 1–40 的完整速度探索历史、Round 19 的历史最好点以及 Round 41 的终止 screen，见
[历史图](docs/performance-views/02-history.svg)和[机器可读数据](docs/performance-data/rounds-01-40.json)。

![原始 Round 1–40 候选观测与累计正式检查最佳值；Round 19 是历史阶段最好点，但不是最终发布版本](docs/performance-views/02-history.svg)

## 实现原理

**两阶段设计**(训练时 GPU 显存里只有 33B transformer 一个大件):

1. **`prepare_cache.py`** —— 离线编码每条训练样本:目标视频 → H3-VisualVAE latent(patch 化行,
   cover-fit 取景——只裁不拉伸)、目标音频 → H3-AudioVAE latent(`--encode-audio`;否则零占位)、
   caption → Qwen3-VL 第 50 层 hidden states。每样本一个 `.pt`。**`prepare_cache_pairs.py`** 是
   长片段变体:把两条连拍片段在时间轴上拼成一条 ~30 秒剧情连续样本(17n+5 重对齐、记录接缝帧、
   双音轨拼接)。
2. **`train.py`** —— 只加载 transformer(`transformer` = FL2VA 变体 / `transformer_ref` = Ref2VA
   变体),用官方 `build_packed_sequence` / `build_row_timesteps` 重建视频/音频/文本打包序列布局,
   优化 rectified-flow MSE loss。

**训练目标。** H3 发布的是 guidance 蒸馏后的 rectified-flow 权重,相对常见的 SD3/Wan 约定有两个
特殊点:

- transformer 的时间输入是 `t = 1 − σ`(t=1 干净、t=0 纯噪);
- transformer 预测的是**指向数据的速度** `v = x₀ − ε`(scheduler 用 `x₀ = x_t + (1−t)·v` 还原,
  注意是**加号**)。

因此 loss 按 `x_t = (1−σ)·x₀ + σ·ε` 加噪,回归目标为 `pred → (x₀ − ε)`;且每步用**两个不同的 σ**:
视频和音频各走一条 shift 曲线(`σ = shift·u / (1+(shift−1)·u)`,视频 shift=12.0、音频 shift=3.0),
共用同一个 `u` 采样——与推理时两个 scheduler 同步推进的 (σ_v, σ_a) 配对分布一致。真实缓存音频以
权重 1.0 进 loss;零占位音频以权重 0 训练——音频头不动,但梯度保持各 rank 同步([FIXES.md](FIXES.md)
第 3 条)。

**验证集。** `md5(sample_id) % 20 == 0`(约 5%)构成确定性 val 划分。每 `--val-every` 步,全 rank
在固定 σ 网格 {0.3, 0.6, 0.9} 上锁步评估,噪声按 (文件, σ) 播种——val 曲线是权重的确定函数,跨步
数、跨 run 可比。`plot_loss.py` 从 `train.log` 画训练/分模态/VAL 曲线。

**可训练参数模式:**

| `--trainable` | 训练什么 | 适用场景 |
|---|---|---|
| `heads` | `proj_out` + `audio_proj_out`(约百万参数) | 单卡冒烟,验证整条链路 |
| `lora`(默认) | PEFT LoRA r=16(`to_qkv`、`to_out.0`、`linear_1`、`linear_2`,约 10.7M 参数) | 单机实用微调 |
| `all` | 全部 33B | 需 `--strategy deepspeed`(附 ZeRO-3 配置) |

`heads`/`lora` 模式的 checkpoint 只保存可训练张量(几 MB 而非 ~66 GB),ZeRO-3 下先从分片聚合再存;
`--resume` 在 `deepspeed.initialize` **之前**以仅权重方式加载(FIXES.md 第 5/9 条)。

## 快速开始

```bash
# 1. 环境:Python 3.11、torch >= 2.8。H3 的类还没进 diffusers 正式发行版——
#    install_env.sh 会安装固定 revision 的集成版本。
bash install_env.sh

# 2. 按 examples/ 里的 schema 写一份 manifest
python validate_manifest.py --manifest path/to/train.jsonl --check-files

# 3. 缓存 latent(debug 规格:256x256、22 帧;正式训练自行调大)
python prepare_cache.py \
  --metadata path/to/train.jsonl \
  --output cache/train \
  --model /path/to/MiniMax-H3 \
  --height 256 --width 256 --frames 22 --encode-text --encode-audio

# 4. 训练(单卡冒烟)
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model /path/to/MiniMax-H3 --variant ref2va \
  --cache cache/train --output runs/smoke \
  --max-steps 10 --trainable heads --strategy ddp

# 4'. 多卡 LoRA(8 卡 DeepSpeed ZeRO-3)——生产路径
deepspeed --num_gpus 8 train.py \
  --model /path/to/MiniMax-H3 --variant ref2va \
  --cache cache/train --output runs/lora \
  --max-steps 800 --trainable lora --strategy deepspeed
```

来自模型本身的约束:`num_frames % 17 == 5`、宽高均为 32 的倍数、视频 24 fps、音频 32 kHz 立体声、
`audio_latent_frames = round(num_frames / 24 * 40)`。

**原始训练器的历史容量记录（8×A800-80GB，并非本次 H20 优化 benchmark）：** 实测上限约 **70k token**/样本
(65k ⇒ 稳态 76 GB;76k 及以上必 OOM)。分辨率 × 时长要按它排布——448×768 @ 27 秒 ≈ 65k 放得下,
576×1024 @ 30 秒 ≈ 127k 放不下。两个旋钮可以在不重编码缓存的前提下执行预算:环境变量
`H3_MAX_LF=<n>` 在加载时把缓存样本截断到 n 个视频 latent 帧;`--max-seq-tokens`(默认 132000)是
前置的全 rank 同步跳过闸门。详见 [FIXES.md](FIXES.md) 第 8 条。

## 效果验证

原始训练器另行完成的 A800 验证表明：数值约定搞错时(timestep 方向 + 速度符号同时反),heads 模式的 loss **越训越高**(10 步内
7.2 → 9.5);用本仓库的修正版,同样配置 loss 稳定在 ≈ 0.3–1.0,8×A800 上 1000 步全程平稳,训出的
输出头 patch 回官方推理管线后生成正常。LoRA 路径在 8×A800 上跑 30 秒 65k-token 序列 + 真实音频,
约 7.5–8 分钟/步(对照:heads 模式 33k token 约 53 秒/步)。详见 [FIXES.md](FIXES.md)。

## 当前局限

- 参考素材尚未编码进条件序列(H3 reference 行):caption 与目标视频/音频参与训练,参考图/参考音频
  还不作为条件。这是实现完整 Ref2VA 微调的主要 TODO。
- epoch 之间无样本 shuffle;batch 恒为每步 1 条序列。
- H3 最终训练阶段使用的稀疏注意力实现未开源,训练只能走全注意力——注意上文 ~70k token 预算。
- `--resume` 仅恢复权重(trainable-only checkpoint 不含优化器状态)。

## 许可

模型权重受 [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
约束(有地域与用途限制;微调产物属 Model Derivatives)。本仓库只含代码——不含模型权重与训练媒体数据。
