import csv
import torch
import torch.distributed as dist

import models
from benchmark.utils import get_tflops, parse_args
from run_benchmark import run_benchmark

# BATCH_SIZE = 128
BATCH_SIZE = 16
SEQ_LEN = 1024

WARMUP = 5
TEST_ITERS = 20

OPTIM = torch.optim.Adam


def build_model():
    model = models.GPT_10B(checkpoint=True).cuda()
    return model


def build_criterion():
    criterion = models.GPTLMLoss()
    return criterion


def get_batch_data():
    input_ids = torch.randint(0, models.VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).cuda()
    mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.int64, device=torch.cuda.current_device())
    return [input_ids, mask], [input_ids]


def main():
    args = parse_args()

    model_size, runtime_stats = run_benchmark(args=args,
                                              model_func=build_model,
                                              loss_func=build_criterion,
                                              batch_data_func=get_batch_data,
                                              warmup_steps=WARMUP,
                                              test_steps=TEST_ITERS,
                                              optim_class=OPTIM)

    if dist.get_rank() == 0:
        with open('log.csv', 'a') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow([
                    'Method', 'Autocast', 'Offload', 'Avg time', 'Max mem allocated', 'Max mem cached', 'TFLOPS',
                    'BParams'
                ])
            writer.writerow([
                args.type,
                str(args.autocast),
                str(args.offload), f'{runtime_stats[0].item():.3f}', f'{runtime_stats[1].item():.2f}',
                f'{runtime_stats[2].item():.2f}',
                f'{get_tflops(model_size, runtime_stats[0].item(), BATCH_SIZE, SEQ_LEN):.2f}', f'{model_size:.3f}'
            ])


if __name__ == '__main__':
    main()
