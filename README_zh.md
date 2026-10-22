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

## 快速开始

准备 Python 3.11、PyTorch ≥ 2.8、公开的 MiniMax-H3 权重，以及符合
[`examples/manifest.example.jsonl`](examples/manifest.example.jsonl) 的 JSONL manifest：

```bash
bash install_env.sh
python validate_manifest.py --manifest path/to/train.jsonl --check-files
python prepare_cache.py \
  --metadata path/to/train.jsonl --output cache/train \
  --model /path/to/MiniMax-H3 \
  --height 256 --width 256 --frames 22 --encode-text --encode-audio

deepspeed --num_gpus 8 train.py \
  --model /path/to/MiniMax-H3 --variant ref2va \
  --cache cache/train --output runs/lora \
  --max-steps 800 --trainable lora --strategy deepspeed
```

优化后的 H20 路径默认启用。缓存预算与 fallback 开关见 `python train.py --help`。模型输入要求包括：
`num_frames % 17 == 5`、宽高为 32 的倍数、24 fps 视频和 32 kHz 立体声音频。

## Agent 优化历史

测试环境为 **8×H20**、BF16 FL2VA LoRA/ZeRO-3、每个 rank batch 为 1、每个样本约 33,750 token。
Round 19 是原始 Round 1–40 中通过完整速度检查的最好结果（约 1.87×），但后来未通过 50-step
轨迹验证，因此不是发布版本。修复版本在 Round 50 首次通过正式速度检查，并在 Round 60 收尾时入选。

![原始 Round 1–40 候选观测与累计正式检查最佳值；Round 19 是历史阶段最好点，但不是最终发布版本](docs/performance-views/02-history.svg)

完整逐轮结果见英文 README 的 [Round 1–40 表](README.md#agent-optimization-history)、
[机器可读数据](docs/performance-data/rounds-01-40.json)和[修复轮次](docs/repair-results.md)。

## 改了什么

- **Attention：**用 cuDNN SDPA 替换 reference attention；在 activation checkpoint 内保留 output/LSE，
  避免 backward 重算；融合 RoPE 与输入布局，减少独立 kernel launch 和 layout 开销。
- **Frozen Linear 缓存：**在每个 rank 的 184 GiB host / 16 GiB GPU 有界 arena 中缓存符合条件的
  frozen GEMM 输出，使用 pinned 的 2 次幂 slab、异步拷贝和预取。
- **RMSNorm 与 replay 正确性：**编译 RMSNorm 加速热点路径；发布修复让 checkpoint inference/replay
  共享同一个 detached compiled Q/K forward，同时保留编译的一阶导数，消除实际观测到的模式相关 Q/K 不一致。
- **保护机制：**不支持的 shape 自动回退原生算子；runtime counter 报告 cache hit、fallback 和峰值分配；
  frozen numerical check 覆盖全部 108 个可训练 LoRA tensor。

这些是累积配置的最终效果，并非各项独立 ablation。详见[性能审计](docs/performance-audit.md)、
[修复轮次](docs/repair-results.md)和[机器可读选择证据](docs/performance-data/repaired-selection.json)。

## 验证：为什么 Round 19 失败、Round 50 通过

原始 Round 19 以 **1.869055×** 通过两步速度检查，但没有通过预先声明的三 seed 轨迹门槛。
seeds 424242、424243、424244 的 reference 和 optimized 共六个 arm 都运行了恰好 50 个 optimizer step，
并在 step 25 和 50 记录 total loss、video loss、audio loss 和全部可训练 LoRA 的 pre-clip 梯度 L2。
optimized arm 出现了间歇性有限梯度尖峰：step 25 的平均梯度 L2 为 **895,829.90**，reference 为
**10.68**，绝对标准化均值差为 **142,950.33**；seed 424243 在 step 18 也出现 **11,248,900.41**
对 **5.95** 的尖峰。梯度裁剪保证更新仍为有限值，但不能让轨迹一致性门槛通过。

定向 replay 诊断随后发现一个明确的 checkpoint 不变量缺陷：8 个 rank 中有 5 个在 Q/K RMSNorm
输入、权重、epsilon 和 stride 完全相同时，因为 inference 与 replay 选择不同 XBLOCK，编译输出仍然
不同。修复让两种 checkpoint 模式共享同一个 detached compiled Q/K forward，并保留既有的编译一阶
导数。它修复了已证实的不一致，但不声称这是所有历史尖峰的唯一原因。

修复候选在 **Round 50** 首次通过正式完整速度检查，并在 Round 60 收尾时入选：**63.176236 秒/step**
对 reference **117.318886 秒/step**，即 **1.8570097438×**。随后以相同三个 seed 重新运行六个 arm、
每个 arm 恰好 50 step；所有值有限、输入/RNG/指纹/契约匹配，8 项固定比较全部满足
`|标准化均值差| ≤ 2.0`：

| 指标 | Step 25 | Step 50 | 门槛 |
|---|---:|---:|---|
| Total loss | -0.031530 | +0.191572 | PASS |
| Video loss | +0.000087 | -0.061845 | PASS |
| Audio loss | -0.033384 | +1.469818 | PASS |
| 全量 LoRA 梯度 L2 | -0.068377 | +0.177239 | PASS |

公开运行时代码 commit 为
[`7d7be36`](https://github.com/2018cx/Faster-MiniMax-H3/commit/7d7be367a2611193b354570a909c8fcc85216394)，
代码树与审计 commit `5c47dee7974a8b69990f05ab4e32cf298bf4ee7e` 一致。完整证据见
[原始失败验证](docs/validation-results.md)、[修复审计](docs/repair-results.md)和
[修复后验证](docs/repaired-validation-results.md)。该验证只证明 50-step 短轨迹一致性，不代表完整收敛
或生成质量。

## 许可

模型权重受 [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
约束(有地域与用途限制;微调产物属 Model Derivatives)。本仓库只含代码——不含模型权重与训练媒体数据。
