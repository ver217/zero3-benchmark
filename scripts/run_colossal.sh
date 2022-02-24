#!/bin/bash


torchrun --standalone --nproc_per_node=8 \
    run_benchmark.py \
    --type ca 


torchrun --standalone --nproc_per_node=8 \
    run_benchmark.py \
    --type ca --autocast


torchrun --standalone --nproc_per_node=8 \
    run_benchmark.py \
    --type ca --offload 


torchrun --standalone --nproc_per_node=8 \
    run_benchmark.py \
    --type ca --autocast --offload
