#!/bin/bash

EXEC_FILE=$1
STAGE=${2:--1}
NPROCS=${3:-8}

EXEC_COMMAND="torchrun --standalone --nproc_per_node=${NPROCS} ${EXEC_FILE} --type ca"

# stage 1
if [[ "$STAGE" -eq 1 || "$STAGE"  -eq -1 ]]
then
    $EXEC_COMMAND --config ./configs/colossalai/stage1.py
    echo "finished benchmarking for Colossal-AI ZeRO Stage 1"
fi

if [[ "$STAGE" == 2 || "$STAGE"  -eq -1 ]]
then
    $EXEC_COMMAND --config ./configs/colossalai/stage2.py
    $EXEC_COMMAND --config ./configs/colossalai/stage2_offload.py
    echo "finished benchmarking for Colossal-AI ZeRO Stage 2"
fi


if [[ "$STAGE" == 3 || "$STAGE"  -eq -1 ]]
then
    $EXEC_COMMAND --config ./configs/colossalai/stage3.py
    $EXEC_COMMAND --config ./configs/colossalai/stage3_offload.py
    $EXEC_COMMAND --config ./configs/colossalai/stage3_autocast.py
    $EXEC_COMMAND --config ./configs/colossalai/stage3_offload_autocast.py
    echo "finished benchmarking for Colossal-AI ZeRO Stage 3"
fi