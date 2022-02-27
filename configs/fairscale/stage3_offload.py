stage = 3
autocast = True
zero = dict(optimizer=dict(),
            model=dict(mixed_precision=autocast,
                       flatten_parameters=False,
                       reshard_after_forward=False,
                       move_params_to_cpu=True,
                       move_grads_to_cpu=True))
