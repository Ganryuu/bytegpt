#!/bin/bash
# Launch the matched-compute H-Net run, resuming from the last checkpoint
# after a crash (up to 8 attempts) until the log shows the FINAL line.
#
# Budget: 1.01e19 FLOPs. At the observed chunking rate N~3.6 the model costs
# ~265M FLOPs/byte (88M byte-rate + 636M per chunk / 3.6), so
# 1.01e19 / 265e6 / 524288 bytes-per-step ~= 72,800 iters.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/hnet-44B
for attempt in $(seq 1 8); do
    RESUME=""
    [ -f $OUT/ckpt.pt ] && RESUME="--resume $OUT/ckpt.pt"
    echo "=== launch attempt $attempt $(date) $RESUME ===" >> train_hnet.log
    .venv-hnet/bin/torchrun --standalone --nproc_per_node=2 hnet_train.py \
        --config configs/hnet_small.json --block-size 8192 \
        --micro-bs 2 --grad-accum 16 \
        --max-iters 72800 --warmup-iters 2000 \
        --eval-every 500 --ckpt-every 1000 --snap-every 8000 \
        --run-name hnet-small-44B --out-dir $OUT $RESUME >> train_hnet.log 2>&1
    grep -q "^FINAL" train_hnet.log && break
    sleep 60
done
