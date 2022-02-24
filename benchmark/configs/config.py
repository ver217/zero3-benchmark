class Config(dict):
    """This is a wrapper class for dict objects so that values of which can be
    accessed as attributes.
    :param config: The dict object to be wrapped
    :type config: dict
    """
    def __init__(self, config: dict = None):
        if config is not None:
            for k, v in config.items():
                self._add_item(k, v)

    def __missing__(self, key):
        raise KeyError(key)

    def __getattr__(self, key):
        try:
            value = super(Config, self).__getitem__(key)
            return value
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        super(Config, self).__setitem__(key, value)

    def _add_item(self, key, value):
        if isinstance(value, dict):
            self.__setattr__(key, Config(value))
        else:
            self.__setattr__(key, value)

    def update(self, config):
        assert isinstance(
            config,
            (Config, dict)), 'can only update dictionary or Config objects.'
        for k, v in config.items():
            self._add_item(k, v)
        return self