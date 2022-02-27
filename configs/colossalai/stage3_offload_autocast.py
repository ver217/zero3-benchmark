stage = 3
optimizer = dict(lr=0.001)
autocast = True
zero = dict(model=dict(mixed_precision=True, reshard_after_forward=False, offload_config=dict(device='cpu')))
