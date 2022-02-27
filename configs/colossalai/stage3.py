stage = 3
optimizer = dict(lr=0.001)
zero = dict(model=dict(
    mixed_precision=True,
    reshard_after_forward=False,
))
