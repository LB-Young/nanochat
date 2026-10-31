#!/bin/bash
set -e
DEST=/root/nanochat/data/base_data_climbmix
mkdir -p $DEST
cd $DEST
BASE=https://modelscope.cn/datasets/AI-ModelScope/climbmix-400b-shuffle/resolve/master
for i in 00000 00001 06542; do
  F="shard_${i}.parquet"
  if [ -s "$F" ]; then echo "skip $F"; continue; fi
  echo "=== downloading $F ==="
  curl -L --noproxy '*' --retry 3 --max-time 300 -o "$F.tmp" "$BASE/$F"
  mv "$F.tmp" "$F"
done
ls -lh $DEST