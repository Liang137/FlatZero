#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

TASK=${TASK:-SST-2}
K=${K:-32}
SEED=${SEED:-42}
BS=${BS:-64}
LR=${LR:-1e-5}
EPS=${EPS:-2e-3}
WD=${WD:-0}
STEP=${STEP:-100000}
EVAL_STEP=${EVAL_STEP:-1000}
MODEL=${MODEL:-roberta-large}

LOGITS=$(jq -n '{"SNLI": 3, "MNLI": 3, "trec": 6, "sst-5": 5}["'$TASK'"] // 2')

GR_TAG=seed$SEED-bs$BS-lr$LR-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP
TAG=${TAG:-k${K}-${MODEL}-zo-flat}
echo "Grid search tag: $GR_TAG"
echo "Tag: $TAG"

TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
    bash examples/run_fewshot.sh \
    --per_device_train_batch_size $BS \
    --learning_rate $LR \
    --eval_steps $EVAL_STEP \
    --weight_decay $WD \
    --zero_order_eps $EPS \
    --zero_order_optim \
    --lr_scheduler_type "constant" \
    --optimizer "sgd" \
    --efficient_zero_order \
    --trace_hessian \
    $@
