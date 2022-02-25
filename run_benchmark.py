
import csv
from functools import partial

import colossalai
import deepspeed
import torch
import torch.distributed as dist
from colossalai.logging import disable_existing_loggers

import benchmark.colossal as ca
import benchmark.deepspeed as ds
import benchmark.fairscale as fs
import models
from benchmark.configs.common import *
from benchmark.utils import get_model_size, get_tflops, get_time, parse_args


def main():
    args = parse_args()

    if args.type == 'ds':
        deepspeed.init_distributed()
    else:
        disable_existing_loggers()
        colossalai.launch_from_torch(config={})
        # dist.init_process_group(backend='nccl')

    torch.cuda.set_device(dist.get_rank())

    if dist.get_rank() == 0:
        print('initialized torch distributed')
        print(vars(args))

    model = models.GPT_10B(checkpoint=True).cuda()
    criterion = models.GPTLMLoss()

    model_size = get_model_size(model)
    if dist.get_rank() == 0:
        print(f'Model size: {model_size:.3f}B')

    if args.type == 'fs':
        model, optimizer = fs.init(model, offload=args.offload)
        run_iter_func = partial(fs.run_iter, autocast=args.autocast)
    elif args.type == 'ds':
        model = ds.init(model, args)
        optimizer = None
        run_iter_func = ds.run_iter
    elif args.type == 'ca':
        model, optimizer = ca.init(model, offload=args.offload)
        run_iter_func = partial(ca.run_iter, autocast=args.autocast)

    for _ in range(WARMUP):
        run_iter_func(model, optimizer, criterion)

    start = get_time()
    for _ in range(TEST_ITERS):
        run_iter_func(model, optimizer, criterion)
    end = get_time()

    stats = torch.tensor([
        (end - start) / TEST_ITERS,
        torch.cuda.max_memory_allocated() / 1024 ** 3,
        torch.cuda.max_memory_cached() / 1024 ** 3
    ], device=torch.cuda.current_device())

    dist.all_reduce(stats)
    stats.div_(dist.get_world_size())

    if dist.get_rank() == 0:
        with open('log.csv', 'a') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['Method', 'Autocast', 'Offload', 'Avg time',
                                'Max mem allocated', 'Max mem cached', 'TFLOPS', 'BParams'])
            writer.writerow([args.type, str(args.autocast), str(args.offload),
                            f'{stats[0].item():.3f}', f'{stats[1].item():.2f}', f'{stats[2].item():.2f}',
                             f'{get_tflops(model_size, stats[0].item(), BATCH_SIZE, SEQ_LEN):.2f}',
                             f'{model_size:.3f}'])


if __name__ == '__main__':
    main()
