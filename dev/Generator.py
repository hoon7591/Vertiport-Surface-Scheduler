from Instance import Instance, InstanceConfig
from Scenario import Scenario, ScenarioConfig


def generate_instance(config: InstanceConfig = InstanceConfig()) -> Instance:
    return Instance.from_config(config)


def generate_scenario_exp(config: ScenarioConfig = ScenarioConfig()) -> Scenario:
    return Scenario.from_scenario_config_exp(config)
