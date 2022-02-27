# ZeRO3 Benchmark

## Run Benchmarking

```shell
# run deepspeed
bash ./scripts/run_deepspeed.sh <benchmark_file> <stage> <num_gpu>

# run fairscale
bash ./scripts/run_fairscale.sh <benchmark_file> <stage> <num_gpu>

# run colossalai
bash ./scripts/run_colossalai.sh <benchmark_file> <stage> <num_gpu>
```

For example, a sample command will be:
```shell
# run for only stage 1 on 8 GPUs
bash ./scripts/run_colossalai.sh run_resnet_benchmark.py 1 8

# run for all stages on 4 GPUs
bash ./scripts/run_colossalai.sh run_resnet_benchmark.py -1 4
```