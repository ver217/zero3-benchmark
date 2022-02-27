import colossalai
import deepspeed
import torch
import torch.distributed as dist
from colossalai.logging import disable_existing_loggers
from typing import Callable

from pprint import pprint
from argparse import Namespace
from benchmark.utils import get_model_size, get_time
from benchmark.colossalai_benchmarker import ColossalaiBenchmarker
from benchmark.deepspeed_benchmarker import DeepSpeedBenchmarker
from benchmark.fairscale_benchmarker import FairScaleBenchmarker


def run_benchmark(args: Namespace, model_func: Callable, loss_func: Callable, batch_data_func: Callable,
                  warmup_steps: int, test_steps: int, optim_class: torch.optim.Optimizer):
    if args.type == 'ds':
        deepspeed.init_distributed()
        torch.cuda.set_device(dist.get_rank())
    else:
        disable_existing_loggers()
        colossalai.launch_from_torch(config={})

    if dist.get_rank() == 0:
        print('initialized torch distributed')
        pprint(vars(args))

    model = model_func()
    criterion = loss_func()

    model_size = get_model_size(model)
    if dist.get_rank() == 0:
        print(f'Model size: {model_size:.3f}B')

    if args.type == 'ca':
        benchmarker = ColossalaiBenchmarker(args.config, optim_class, model, criterion)
    elif args.type == 'ds':
        benchmarker = DeepSpeedBenchmarker(args, model, criterion)
    elif args.type == 'fs':
        benchmarker = FairScaleBenchmarker(args.config, optim_class, model, criterion)

    for _ in range(warmup_steps):
        data, label = batch_data_func()
        benchmarker.run_iter(data=data, label=label)

    start = get_time()
    for _ in range(test_steps):
        data, label = batch_data_func()
        benchmarker.run_iter(data=data, label=label)
    end = get_time()

    runtime_stats = torch.tensor([(end - start) / test_steps,
                                  torch.cuda.max_memory_allocated() / 1024**3,
                                  torch.cuda.max_memory_cached() / 1024**3],
                                 device=torch.cuda.current_device())

    dist.all_reduce(runtime_stats)
    runtime_stats.div_(dist.get_world_size())

    return model_size, runtime_stats
