stage = 1
optimizer = dict(lr=0.001)
autocast = True
zero = dict(optimizer=dict(broadcast_fp16=autocast))
