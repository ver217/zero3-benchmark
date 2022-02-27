#!/bin/bash

EXEC_FILE=$1
STAGE=${2:--1}
NPROCS=${3:-8}

bash ./scripts/run_colossal.sh $EXEC_FILE $STAGE $NPROCS
bash ./scripts/run_deepspeed.sh $EXEC_FILE $STAGE $NPROCS
bash ./scripts/run_fairscale.sh $EXEC_FILE $STAGE $NPROCS
