#!/bin/bash

EXEC_FILE=$1
STAGE=${2:--1}
NPROCS=${3:-8}

EXEC_COMMAND="torchrun --standalone --nproc_per_node=${NPROCS} ${EXEC_FILE} --type fs"

# stage 1
if [[ "$STAGE" -eq 1 || "$STAGE"  -eq -1 ]]
then
    $EXEC_COMMAND --config ./configs/fairscale/stage1_autocast.py
    echo "finished benchmarking for FairScale ZeRO Stage 1"
fi

if [[ "$STAGE" == 2 || "$STAGE"  -eq -1 ]]
then
    $EXEC_COMMAND --config ./configs/fairscale/stage2_autocast.py
    echo "finished benchmarking for FairScale ZeRO Stage 2"
fi

if [[ "$STAGE" == 3 || "$STAGE"  -eq -1 ]]
then
    $EXEC_COMMAND --config ./configs/fairscale/stage3_autocast.py
    $EXEC_COMMAND --config ./configs/fairscale/stage3_offload_autocast.py
    echo "finished benchmarking for FairScale ZeRO Stage 3"
fi
