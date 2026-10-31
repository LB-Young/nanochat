import os, json, random
import pyarrow as pa
import pyarrow.parquet as pq

BASE = "/root/nanochat/data/task_data"
dst_root = os.path.join(BASE, "cais--mmlu", "all")

rng = random.Random(0)
subjects = ["high_school_math", "college_physics", "abstract_algebra"]

for split, n in [("auxiliary_train", 10000), ("test", 6000)]:
    dst = os.path.join(dst_root, split)
    # wipe existing
    for f in os.listdir(dst):
        os.remove(os.path.join(dst, f))
    rows_q, rows_c, rows_a, rows_s = [], [], [], []
    for i in range(n):
        rows_q.append(f"Sample question {i}: what is 2+{i%5}?")
        rows_c.append([f"option {j}" for j in range(4)])
        rows_a.append(rng.randint(0, 3))
        rows_s.append(rng.choice(subjects))
    t = pa.table({
        "question": rows_q,
        "choices": pa.array(rows_c, type=pa.list_(pa.string())),
        "answer": rows_a,
        "subject": rows_s,
    })
    pq.write_table(t, os.path.join(dst, "00000.parquet"))
    with open(os.path.join(dst, "manifest.json"), "w") as f:
        json.dump(["00000.parquet"], f)
    print(f"mmlu {split}: {n} rows")
