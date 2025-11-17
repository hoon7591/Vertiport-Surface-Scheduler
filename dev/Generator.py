from Instance import Instance, InstanceConfig
from Scenario import Scenario, ScenarioRHCConfig


def generate_instance(config: InstanceConfig = InstanceConfig()) -> Instance:
    return Instance.from_config(config)


def generate_scenario_exp(config: ScenarioRHCConfig = ScenarioRHCConfig()) -> Scenario:
    return Scenario.from_scenario_config_exp(config)
