stage = 1
optimizer = dict(lr=0.001)
zero = dict(optimizer=dict(partition_grad=False, overlap_communication=True))
