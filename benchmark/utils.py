import argparse
import time

import torch

import deepspeed


def get_time():
    torch.cuda.synchronize()
    return time.time()


def get_tflops(num_params_in_b: float, iter_time: float, batch_size: int, seq_len: int) -> float:
    gflops = num_params_in_b * batch_size * seq_len * 2.0 * 4.0
    return gflops / (iter_time * 1000.0)


def get_model_size(model: torch.nn.Module, unit: str = 'B') -> float:
    assert unit in ('B', 'M')
    num_params = sum(p.numel() for p in model.parameters())
    if unit == 'B':
        return num_params / 1024**3
    return num_params / 1024**2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, choices=['ds', 'fs', 'ca'])
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--autocast', default=False, action='store_true')
    parser.add_argument('--offload', default=False, action='store_true')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args
