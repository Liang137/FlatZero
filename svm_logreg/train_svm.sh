#!/bin/bash

TASK=${TASK:-a5a}
LR=${LR:-1e-4}
LAM=${LAM:-0.05}
ALGO=${ALGO:-zo}
STEP=${STEP:-10000}
SEED=${SEED:-29}

python svm.py \
    --lr $LR \
    --lam $LAM \
    --algo $ALGO \
    --data $TASK \
    --max_iter $STEP \
    --seed $SEED