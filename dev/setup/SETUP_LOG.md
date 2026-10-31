# nanochat WSL 部署与跑通记录（Runbook）

> 环境：WSL2 Ubuntu-24.04，2× RTX 5060 Ti 16GB（Blackwell sm_120）
> 代码位置：`/root/nanochat`
> 数据/checkpoint 根目录：`/root/nanochat/data/`
> 配套脚本：本目录（`dev/setup/`）下的 `dl_*.sh/py`、`layout_*.py`、`mk_mmlu*.py`

---

## 0. 每次开新终端先做的事

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate nanochat
cd /root/nanochat

export NANOCHAT_BASE_DIR=/root/nanochat/data
export OMP_NUM_THREADS=1
export WANDB_RUN=dummy     # dummy = 不上 wandb，纯本地
```

---

## 1. 从零部署（已做过，不用重复）

### 1.1 创建 conda 环境

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda create -y -n nanochat python=3.11
conda activate nanochat
```

### 1.2 装依赖

```bash
# torch 必须 cu128（Blackwell sm_120 只有 cu128 wheel 支持）
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1

pip install filelock numpy psutil pyarrow requests tiktoken rustbpe kernels \
            wandb python-dotenv matplotlib ipykernel pytest modelscope
```

验证：
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# 应输出: 2.9.1+cu128 True 2
```

---

## 2. 下载数据（全部走 modelscope 国内镜像）

### 2.1 预训练数据（climbmix-400b-shuffle）

nanochat 原生从 huggingface.co 下，国内不通。用 modelscope 镜像 `AI-ModelScope/climbmix-400b-shuffle`。

**脚本：`dev/setup/dl.sh`（smoke test 3 个 shard）、`dev/setup/dl_climbmix.sh`（20GB / 230 个 shard）**

```bash
# smoke test：3 个 shard，每个 ~88MB
mkdir -p /root/nanochat/data/base_data_climbmix
BASE=https://modelscope.cn/datasets/AI-ModelScope/climbmix-400b-shuffle/resolve/master

for i in 00000 00001 06542; do
  curl -L --noproxy '*' --retry 3 --max-time 300 \
       -o /root/nanochat/data/base_data_climbmix/shard_${i}.parquet \
       "$BASE/shard_${i}.parquet"
done
```

> 真训要更多 shard 时，直接跑 `bash dev/setup/dl_climbmix.sh`（下 shard 0~230，约 20GB），最后记得 06542（验证集）已含。

### 2.2 SFT 数据集

SFT 需要 SmolTalk / GSM8K / MMLU，代码里写死从 HF API 拉，国内不通。手动下到代码期望的布局。

代码期望布局：`$NANOCHAT_BASE_DIR/task_data/<org>--<name>/<subset>/<split>/00000.parquet` + `manifest.json`

**下载脚本：**
- `dev/setup/dl_sm.py` — 下 SmolTalk（`AI-ModelScope/smoltalk`，完整版 3.9GB）
- `dev/setup/dl_misc.py` — 下 GSM8K（`AI-ModelScope/gsm8k`）+ 尝试 MMLU
- `dev/setup/dl_mmlu.py` — 下 MMLU（`modelscope/mmlu`，老 CSV 格式，与 cais/mmlu 不兼容，仅供参考）

**布局脚本（把下载好的数据整理成代码期望的目录结构）：**
- `dev/setup/layout_sm.py` — SmolTalk 布局
- `dev/setup/layout_gs.py` — GSM8K 布局
- `dev/setup/mk_mmlu.py` / `mk_mmlu2.py` — 合成最小可用 MMLU parquet（字段：question, choices, answer, subject）

> ⚠️ `--noproxy '*'` 必须加：WSL 里有 `https_proxy=http://127.0.0.1:7897`（Clash），但该代理节点对 huggingface.co / google.com 都 TLS 握手失败；modelscope 是国内站，直连 20MB/s。

---

## 3. 训练与推理

### 3.1 训 tokenizer

```bash
python -m scripts.tok_train --max-chars 50000000
# 50M chars ~2 秒训完，产物在 $NANOCHAT_BASE_DIR/tokenizer/
```

### 3.2 预训练 base model

```bash
torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
    --depth=12 \
    --num-iterations=20 \
    --device-batch-size=8 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --eval-every=10 \
    --target-param-data-ratio=-1 \
    --run=dummy
```

实测：20 步，loss 10.4→7.4，val bpb 3.18→1.90，~65k tok/s，峰值显存 10.1GB。
checkpoint 落在 `$NANOCHAT_BASE_DIR/base_checkpoints/d12/`。

### 3.3 SFT

```bash
torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft -- \
    --num-iterations=10 \
    --device-batch-size=4 \
    --eval-every=5 \
    --chatcore-every=-1 \
    --run=dummy \
    --model-tag=d12 \
    --model-step=20 \
    --mmlu-epochs=1 \
    --gsm8k-epochs=1
```

checkpoint 落在 `$NANOCHAT_BASE_DIR/chatsft_checkpoints/d12/`。

### 3.4 推理

单轮提问：
```bash
python -m scripts.chat_cli -g d12 -s 1 -p "Why is the sky blue?"
```

交互式对话：
```bash
python -m scripts.chat_cli -g d12 -s 1
```

> `-g` = model-tag（checkpoint 目录名），`-s` = step（model_00000X.pt 的 X）。

---

## 4. 踩过的坑 & 解决方案

1. **conda 在哪**：装在 `/opt/miniconda3`，不是 `/root/miniconda3`。非登录 shell 里要先 `source /opt/miniconda3/etc/profile.d/conda.sh`。
2. **Blackwell 必须 cu128**：RTX 5060 Ti 是 sm_120，cu121/cu124 的 torch 会报 `no kernel image is available`。用 `--index-url https://download.pytorch.org/whl/cu128`。
3. **HF 直连不通**：huggingface.co 和 hf-mirror.com 在你当前 Clash 代理下 TLS 都被重置。全部数据走 modelscope 国内直连，curl 加 `--noproxy '*'` 绕开 Clash。
4. **dataset.py 间接 import torch**：torch 没装完就跑 `python -m nanochat.dataset` 会报 `No module named 'torch'`。先装完 torch 再下数据。
5. **MMLU 行数不够会越界**：`chat_sft.py` 里 val 集写死 `MMLU(test, stop=5200)`，所以 MMLU test 至少要 6000 行、auxiliary_train 至少 10000 行。
6. **`AI-ModelScope/smol-smoltalk` 是空壳**：API 404。用 `AI-ModelScope/smoltalk`（完整版，`data/all/` 子集）。
7. **chat_cli 参数是短参数**：`-g tag -s step`，不是 `--model-tag/--model-step`。
8. **16GB 显存要降 batch-size**：speedrun.sh 默认 device-batch-size=16 是给 8×H100 的。d12 模型实测预训练=8、SFT=4 刚好。

---

## 5. 真训时改什么

- `--depth 12` → `16` 或 `24`（GPT-2 能力约 d24）
- `--num-iterations 20` → 几千~几万
- 预训练 shard：`for i in 00000 00001 06542` → 多下到 230 个（`bash dev/setup/dl_climbmix.sh`）
- 开 wandb：去掉 `WANDB_RUN=dummy`，先 `wandb login`
- 显存不够就 `--device-batch-size` 从 8 降到 4/2
