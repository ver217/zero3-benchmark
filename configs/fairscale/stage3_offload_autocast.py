stage = 3
autocast = True
offload = True
zero = dict(optimizer=dict(),
            model=dict(mixed_precision=autocast,
                       flatten_parameters=False,
                       reshard_after_forward=False,
                       move_params_to_cpu=offload,
                       move_grads_to_cpu=offload))
