#!/bin/bash
# Download ~20GB of climbmix shards from modelscope
set -u
DEST=/root/nanochat/data/base_data_climbmix
mkdir -p "$DEST"
BASE=https://modelscope.cn/datasets/AI-ModelScope/climbmix-400b-shuffle/resolve/master

# shard 00000..0230 = 231 shards * ~88MB 鈮?20GB
# 06542 (val) already downloaded earlier
download_one() {
  local i=$1
  local f=$(printf "shard_%05d.parquet" $i)
  local path="$DEST/$f"
  if [ -s "$path" ]; then
    echo "skip $f"
    return
  fi
  curl -sL --noproxy '*' --retry 3 --max-time 300 -o "$path.tmp" "$BASE/$f" \
    && mv "$path.tmp" "$path" \
    && echo "ok $f $(du -h "$path" | cut -f1)" \
    || echo "FAIL $f"
}
export -f download_one
export DEST BASE

seq 0 230 | xargs -P 8 -I{} bash -c 'download_one "$@"' _ {}

echo "=== done ==="
ls -lh "$DEST" | tail -5
du -sh "$DEST"
