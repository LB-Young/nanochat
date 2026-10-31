import os, json, shutil
import pyarrow.parquet as pq

BASE = "/root/nanochat/data/task_data"

# GSM8K: openai/gsm8k, subset=main
# mirror layout: /root/gsm8k-mirror/main/train-00000-of-00001.parquet, test-00000-of-00001.parquet
dst_root = os.path.join(BASE, "openai--gsm8k", "main")
for split, src_name in [("train", "train-00000-of-00001.parquet"),
                        ("test", "test-00000-of-00001.parquet")]:
    dst = os.path.join(dst_root, split)
    os.makedirs(dst, exist_ok=True)
    src = os.path.join("/root/gsm8k-mirror/main", src_name)
    shutil.copy(src, os.path.join(dst, "00000.parquet"))
    with open(os.path.join(dst, "manifest.json"), "w") as f:
        json.dump(["00000.parquet"], f)
    t = pq.read_table(os.path.join(dst, "00000.parquet"))
    print(f"gsm8k {split}: cols={t.column_names} rows={len(t)}")
