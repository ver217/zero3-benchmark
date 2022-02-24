#!/bin/bash


torchrun --standalone --nproc_per_node=8 \
    run_benchmark.py \
    --type fs 


torchrun --standalone --nproc_per_node=8 \
    run_benchmark.py \
    --type fs --autocast


torchrun --standalone --nproc_per_node=8 \
    run_benchmark.py \
    --type fs --offload 


torchrun --standalone --nproc_per_node=8 \
    run_benchmark.py \
    --type fs --autocast --offload
