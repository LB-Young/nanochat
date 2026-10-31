import os, json
import pyarrow as pa
import pyarrow.parquet as pq

BASE = "/root/nanochat/data/task_data"
# cais/mmlu, subset="all"
dst_root = os.path.join(BASE, "cais--mmlu", "all")

# Build a tiny synthetic MMLU-like table.
# cais/mmlu "all" fields: question (string), choices (list<item: string>), answer (int64), subject (string)
# We only need question/choices/answer for the code.

def make_rows(n, seed=0):
    import random
    rng = random.Random(seed)
    rows = []
    subjects = ["high_school_math", "college_physics", "abstract_algebra"]
    for i in range(n):
        q = f"Sample question {i}: what is 2+{i%5}?"
        choices = [f"option {j}" for j in range(4)]
        ans = rng.randint(0, 3)
        rows.append({"question": q, "choices": choices, "answer": ans,
                      "subject": rng.choice(subjects)})
    return rows

for split, n in [("auxiliary_train", 500), ("test", 500)]:
    dst = os.path.join(dst_root, split)
    os.makedirs(dst, exist_ok=True)
    rows = make_rows(n)
    t = pa.table({
        "question": [r["question"] for r in rows],
        "choices": pa.array([r["choices"] for r in rows], type=pa.list_(pa.string())),
        "answer": [r["answer"] for r in rows],
        "subject": [r["subject"] for r in rows],
    })
    pq.write_table(t, os.path.join(dst, "00000.parquet"))
    with open(os.path.join(dst, "manifest.json"), "w") as f:
        json.dump(["00000.parquet"], f)
    print(f"mmlu {split}: {n} rows written")
