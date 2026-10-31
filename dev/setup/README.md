# dev/setup — 数据下载与数据布局脚本

这些脚本是本机（WSL Ubuntu-24.04 + modelscope 国内镜像）跑通 nanochat 全流程时写下的辅助脚本。
因为 HF 在国内不通，全部数据走 modelscope 直连下载，再按 nanochat 代码期望的布局整理到 `$NANOCHAT_BASE_DIR/task_data/`。

> 所有脚本应在 **WSL 内**执行（`source /opt/miniconda3/etc/profile.d/conda.sh && conda activate nanochat` 之后），
> 下载类命令一律带 `--noproxy '*'` 绕开 Clash 代理。

## 文档

- **`SETUP_LOG.md`** — 完整 runbook：部署 → 数据 → tokenizer → 预训练 → SFT → 推理，含踩坑记录（9 个坑及解法）。

## 预训练数据（climbmix-400b-shuffle，modelscope 镜像 `AI-ModelScope/climbmix-400b-shuffle`）

| 脚本 | 作用 |
|---|---|
| `dl.sh` | 下载 smoke test 用的 3 个 shard（00000 / 00001 训练 + 06542 验证，~264MB） |
| `dl_climbmix.sh` | 8 并发下载 231 个训练 shard（00000~00230）+ 验证 06542，约 20GB，断点续传（已存在则跳过） |

## SFT 数据

| 脚本 | 作用 |
|---|---|
| `dl_sm.py` | modelscope SDK 下载 `AI-ModelScope/smoltalk`（完整版 3.9GB，含 14 个子集）到 `/root/smoltalk-mirror` |
| `dl_smol.py` | （失败尝试）`AI-ModelScope/smol-smoltalk` 是空壳 404，勿用 |
| `dl_misc.py` | 下载 `AI-ModelScope/gsm8k` 到 `/root/gsm8k-mirror`，并尝试 MMLU |
| `dl_mmlu.py` | 下载 `modelscope/mmlu`（159MB），老 CSV tar 格式，与 `cais/mmlu` parquet schema 不兼容，仅参考 |
| `layout_sm.py` | SmolTalk → `task_data/HuggingFaceTB--smol-smoltalk/default/{train,test}/` + manifest.json |
| `layout_gs.py` | GSM8K → `task_data/openai--gsm8k/main/{train,test}/` + manifest.json |
| `mk_mmlu.py` / `mk_mmlu2.py` | 用 pyarrow 合成最小可用 MMLU parquet（auxiliary_train 10000 行 / test 6000 行，字段 question/choices/answer/subject），满足 `chat_sft.py` 里 `MMLU(test, stop=5200)` 的要求 |

## 代码期望的数据布局（勿改动）

```
$NANOCHAT_BASE_DIR/task_data/
  HuggingFaceTB--smol-smoltalk/default/train/00000.parquet (+ manifest.json)
  HuggingFaceTB--smol-smoltalk/default/test/00000.parquet
  openai--gsm8k/main/train/00000.parquet
  openai--gsm8k/main/test/00000.parquet
  cais--mmlu/all/auxiliary_train/00000.parquet
  cais--mmlu/all/test/00000.parquet
```

## 数据/checkpoint 位置（不进 git）

- `/root/nanochat/data/base_data_climbmix/` — 20GB climbmix parquet
- `/root/nanochat/data/task_data/` — SFT 数据布局
- `/root/nanochat/data/tokenizer/` — 训好的 BPE tokenizer
- `/root/nanochat/data/base_checkpoints/` / `chatsft_checkpoints/` — 模型 checkpoint
- `/root/smoltalk-mirror` / `/root/gsm8k-mirror` / `/root/mmlu-mirror` — 原始下载
