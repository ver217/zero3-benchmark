stage = 2
optimizer = dict(lr=0.001)
autocast = True
zero = dict(optimizer=dict(broadcast_fp16=autocast), model=dict(reduce_fp16=autocast))
