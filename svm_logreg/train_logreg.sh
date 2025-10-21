#!/bin/bash

TASK=${TASK:-a5a}
LR=${LR:-0.01}
LAM=${LAM:-0.1}
ALGO=${ALGO:-zo}
STEP=${STEP:-10000}
SEED=${SEED:-29}

python logreg.py \
    --lr $LR \
    --lam $LAM \
    --algo $ALGO \
    --data $TASK \
    --max_iter $STEP \
    --seed $SEED