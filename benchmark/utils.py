import argparse
import time

import torch

import deepspeed


def get_time():
    torch.cuda.synchronize()
    return time.time()


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
