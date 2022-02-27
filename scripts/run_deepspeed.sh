#!/bin/bash

EXEC_FILE=$1
STAGE=${2:--1}

EXEC_COMMAND="deepspeed --master_port 29550 ${EXEC_FILE} --type ds --deepspeed"

# stage 1
if [[ "$STAGE" -eq 1 || "$STAGE"  -eq -1 ]]
then
    $EXEC_COMMAND --deepspeed_config ./configs/deepspeed/stage1.json
    echo "finished benchmarking for DeepSpeed ZeRO Stage 1"
fi

if [[ "$stage" == 2 || "$STAGE"  -eq -1 ]]
then
    $EXEC_COMMAND --deepspeed_config ./configs/deepspeed/stage2.json
    $EXEC_COMMAND --deepspeed_config ./configs/deepspeed/stage2_offload.json
    echo "finished benchmarking for DeepSpeed ZeRO Stage 2"
fi

if [[ "$stage" == 3 || "$STAGE"  -eq -1 ]]
then
    $EXEC_COMMAND --deepspeed_config ./configs/deepspeed/stage3.json
    $EXEC_COMMAND --deepspeed_config ./configs/deepspeed/stage3_offload.json
    echo "finished benchmarking for DeepSpeed ZeRO Stage 3"
fi