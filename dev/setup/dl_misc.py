from modelscope import dataset_snapshot_download
import sys

for name, local in [
    ("AI-ModelScope/mmlu", "/root/mmlu-mirror"),
    ("AI-ModelScope/gsm8k", "/root/gsm8k-mirror"),
]:
    try:
        print(f"=== downloading {name} ===")
        dataset_snapshot_download(name, local_dir=local)
        print(f"done {name}")
    except Exception as e:
        print(f"FAIL {name}: {e}")
