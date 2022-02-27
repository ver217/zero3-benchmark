stage = 2
optimizer = dict(lr=0.001)
zero = dict(optimizer=dict(partition_grad=True, overlap_communication=True, cpu_offload=True, verbose=True))
