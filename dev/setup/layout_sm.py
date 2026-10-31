import os, json, shutil
import pyarrow.parquet as pq

BASE = "/root/nanochat/data/task_data"

# --- SmolTalk: we have the "smoltalk" full version mirror; code wants "smol-smoltalk" ---
src_train = "/root/smoltalk-mirror/data/all"
dst_dir = os.path.join(BASE, "HuggingFaceTB--smol-smoltalk", "default")

# train: use first 2 shards
train_dst = os.path.join(dst_dir, "train")
os.makedirs(train_dst, exist_ok=True)
train_files = []
for i in range(2):
    src = os.path.join(src_train, f"train-{i:05d}-of-00009.parquet")
    dst = os.path.join(train_dst, f"{i:05d}.parquet")
    shutil.copy(src, dst)
    train_files.append(f"{i:05d}.parquet")
with open(os.path.join(train_dst, "manifest.json"), "w") as f:
    json.dump(train_files, f)
print(f"smoltalk train: {len(train_files)} shards -> {train_dst}")

# test: use the single test shard
test_dst = os.path.join(dst_dir, "test")
os.makedirs(test_dst, exist_ok=True)
src_test = os.path.join(src_train, "test-00000-of-00001.parquet")
shutil.copy(src_test, os.path.join(test_dst, "00000.parquet"))
with open(os.path.join(test_dst, "manifest.json"), "w") as f:
    json.dump(["00000.parquet"], f)
print(f"smoltalk test: 1 shard -> {test_dst}")

# quick sanity: check columns
t = pq.read_table(os.path.join(train_dst, "00000.parquet"))
print("smoltalk columns:", t.column_names, "rows:", len(t))
