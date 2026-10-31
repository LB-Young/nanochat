from modelscope import dataset_snapshot_download
import os, json, shutil
import pyarrow.parquet as pq

# 1) download mmlu
print("=== downloading modelscope/mmlu ===")
try:
    dataset_snapshot_download("modelscope/mmlu", local_dir="/root/mmlu-mirror")
    print("mmlu done")
except Exception as e:
    print(f"mmlu FAIL: {e}")

# list what we got
for root, dirs, files in os.walk("/root/mmlu-mirror"):
    for f in files:
        if f.endswith((".parquet", ".csv", ".json")):
            p = os.path.join(root, f)
            print(p, os.path.getsize(p))
