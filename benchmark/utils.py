import colossalai
import time
import torch
import deepspeed

from colossalai.context import Config
import json

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
    parser = colossalai.get_default_parser()
    parser.add_argument('--type', type=str, choices=['ds', 'fs', 'ca'])

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args

def get_stage(args):
    if args.config:
        stage = Config.from_file(args.config).stage
    else:
        with open(args.deepspeed_config) as f:
            js_config = json.load(f)
            stage = js_config['zero_optimization']['stage']

    return stage

def get_autocast_state(args):
    if args.config:
        return 'autocast' in args.config
    else:
        return 'autocast' in args.deepspeed_config

def get_offload_state(args):
    if args.config:
        return 'offload' in args.config
    else:
        return 'offload' in args.deepspeed_config

