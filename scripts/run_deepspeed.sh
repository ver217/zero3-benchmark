#!/bin/bash

deepspeed run_benchmark.py \
        --type ds \
        --deepspeed \
        --deepspeed_config benchmark/configs/deepspeed_stage3.json

deepspeed run_benchmark.py \
        --type ds \
        --offload \
        --deepspeed \
        --deepspeed_config benchmark/configs/deepspeed_stage3_offload.json