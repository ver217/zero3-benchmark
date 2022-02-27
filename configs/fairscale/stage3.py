autocast = True
zero = dict(optimizer=dict(),
            model=dict(
                mixed_precision=autocast,
                flatten_parameters=False,
                reshard_after_forward=False,
            ))
