from Instance import Instance, InstanceConfig


def generate_instance(config: InstanceConfig = InstanceConfig()) -> Instance:
    return Instance.from_config(config)
