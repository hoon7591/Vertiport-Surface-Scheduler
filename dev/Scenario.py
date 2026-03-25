from typing import Any, List
import numpy as np
from dataclasses import dataclass, field
from Instance import Instance
import copy


@dataclass
class ScenarioRHCConfig:
    """Configuration class for generating an instance of the UAM-VS-Scheduling Problem.
    This class holds the parameters required to create a problem instance, including
    the number of operations, vehicles, pads, gates, and various processing times.
    """
    seed: int = 42
    num_operations: int = 5
    num_vehicles_per_hour: int = 15
    num_pad: int = 2
    num_buffer_in: Any = None
    num_gate: int = 10
    num_buffer_out: Any = None # not used for is_unified_buffer = True: if True, this is merged with buffer_in #TODO : revise implicitly
    num_buffer: Any = None # only used for is_unified_buffer = True
    objective_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])
    proc_air_v: List[float] = field(default_factory=lambda: [2.0, 2.2, 2.8, 3.0])
    proc_air_r: List[float] = field(default_factory=lambda: [0.8, 0.9, 1.0, 1.0, 0.8, 0.9, 1.0, 1.0])
    proc_air_o: List[float] = field(default_factory=lambda: [1.2, 1.0])
    proc_gate_v: List[float] = field(default_factory=lambda: [15.0, 17.0, 23.0, 25.0])
    st_list_v: Any = field(default_factory=lambda: [
        [1.25, 1.0, 1.0, 1.0],
        [1.5, 1.25, 1.0, 1.0],
        [1.75, 1.5, 1.25, 1.0],
        [2.0, 1.75, 1.5, 1.25]
    ])
    st_list_o: List[float] = field(default_factory=lambda: [0.8, 0.6, 0.5, 1.0])
    st_list_r: List[float] = field(default_factory=lambda: [1.0, 0.7, 0.8, 1.0, 1.0, 0.7, 0.8, 1.0])
    ready_max: float = 100.0
    ETA_ready_diff: List[float] = field(default_factory=lambda: [5.0, 5.0, 5.0])  # [lower, upper, mode] of beta distribution; for deterministic case, use [c, c, c]
    lambda_beta_pert: float = 4.0
    ETD_margin: float = 5.0
    gate_close_margin: float = 3.0
    is_unified_buffer: bool = False

    # for receding horizon control
    scheduling_horizon_length: float = 50.0  # in minutes
    update_interval: float = 1.0  # in minutes
    coe_bridge_from_update_interval: float = 0.5  # adjust ratio of update interval as a std of stochastic bridge
    operation_hour: int = 18
    disturbance_proc: Any = field(default_factory=lambda: [
        [-0.2, 0.6, 0.0],
        [-0.3, 1.0, 0.0],
        [-0.2, 0.6, 0.0]
    ])  # [lower, upper, mode] of beta distribution; for deterministic case, use [c, c, c] for each sublist
    disturbance_ready: List[float] = field(default_factory=lambda: [-3.0, 10.0, 1.0])  # [lower, upper, mode] of beta distribution; for deterministic case, use [c, c, c]
    dynamic_arrival_v_id: List[int] = field(default_factory=lambda: [])
    dynamic_arrival_aware_time_range: List[float] = field(default_factory=lambda: [5.0, 20.0])   # lower bound should be larger than update_interval
    dynamic_proc_v_id: List[int] = field(default_factory=lambda: [])
    dynamic_proc_op: List[int] = field(default_factory=lambda: [])
    dynamic_proc_inc_time: List[float] = field(default_factory=lambda: [])          # specifically defined by user
    uncertainty_compensation_for_early_ready: float = 20.0  # usually same as upper bound of dynamic_arrival_aware_time_range


class Scenario:
    def __init__(
        self,
        seed: int,
        num_operations: int,
        num_vehicles_per_hour: int,
        num_vehicles: int,
        vehicle_id: np.ndarray,
        num_pad: int,
        num_buffer_in: int,
        num_gate: int,
        num_buffer_out: int,
        num_buffer: Any,
        num_resource: int,
        objective_weights: List[float],
        proc_air_v: List[float],
        proc_air_r: List[float],
        proc_air_o: List[float],
        proc_gate_v: List[int],
        st_list: Any,
        maximum_arrival_time: float,
        ETA_ready_diff: List[float],
        lambda_beta_pert: float,
        ETD_margin: float,
        gate_close_margin: float,
        is_unified_buffer: bool,
        vehicle_arrival_times: np.ndarray,
        vehicle_arrival_times_init: np.ndarray,
        proc: list,
        first_activated_time_of_ready: np.ndarray,
        vehicle_planned_arrival_times: np.ndarray,
        vehicle_planned_departure_times: np.ndarray,
        vehicle_planned_gate_close_times: np.ndarray,
        ST: dict,
        vehicle_type: np.ndarray,
        M: float,
        whether_vehicle_dynamic_arrival: np.ndarray,
        vehicle_dynamic_arrival_aware_time: np.ndarray,
        proc_real: list,
        coe_bridge_from_update_interval: float,
    ):
        self.seed = seed
        self.num_operations = num_operations
        self.num_vehicles_per_hour = num_vehicles_per_hour
        self.num_vehicles = num_vehicles
        self.vehicle_id = vehicle_id
        self.num_pad = num_pad
        self.num_buffer_in = num_buffer_in
        self.num_gate = num_gate
        self.num_buffer_out = num_buffer_out
        self.num_buffer = num_buffer
        self.num_resource = num_resource
        self.objective_weights = objective_weights
        self.proc_air_v = proc_air_v
        self.proc_air_r = proc_air_r
        self.proc_air_o = proc_air_o
        self.proc_gate_v = proc_gate_v
        self.st_list = st_list
        self.maximum_arrival_time = maximum_arrival_time
        self.ETA_ready_diff = ETA_ready_diff # TODO : check if it needs?
        self.lambda_beta_pert = lambda_beta_pert
        self.ETD_margin = ETD_margin # TODO : check if it needs?
        self.gate_close_margin = gate_close_margin
        self.is_unified_buffer = is_unified_buffer
        self.vehicle_arrival_times = vehicle_arrival_times
        self.vehicle_arrival_times_init = vehicle_arrival_times_init
        self.proc = proc
        self.first_activated_time_of_ready = first_activated_time_of_ready
        self.vehicle_planned_arrival_times = vehicle_planned_arrival_times
        self.vehicle_planned_departure_times = vehicle_planned_departure_times
        self.vehicle_planned_gate_close_times = vehicle_planned_gate_close_times
        self.ST = ST
        self.vehicle_type = vehicle_type
        self.big_M = M
        self.whether_vehicle_dynamic_arrival = whether_vehicle_dynamic_arrival
        self.vehicle_dynamic_arrival_aware_time = vehicle_dynamic_arrival_aware_time
        self.proc_real = proc_real
        self.coe_bridge_from_update_interval = coe_bridge_from_update_interval

        # 🔹 stochastic bridge 전용 난수 생성기
        self.bridge_rng = np.random.default_rng(self.seed + 10000)

    @classmethod
    def from_scenario_config_exp(cls, config: "ScenarioRHCConfig") -> "Scenario":
        np.random.seed(config.seed)
        num_vehicles = config.num_vehicles_per_hour * config.operation_hour
        vehicle_id = np.arange(num_vehicles)

        if config.is_unified_buffer:
            num_resource = config.num_pad + config.num_buffer + config.num_gate
        else:
            num_resource = config.num_pad + config.num_buffer_in + config.num_gate + config.num_buffer_out

        if config.is_unified_buffer:
            if config.num_buffer is None or config.num_buffer == 0:
                config.num_operations = 3
            else:
                config.num_operations = 5
        else:
            if config.num_buffer_in is None or config.num_buffer_in == 0:
                if config.num_buffer_out is None or config.num_buffer_out == 0:
                    config.num_operations = 3
                else:
                    config.num_operations = 4
            else:
                if config.num_buffer_out is None or config.num_buffer_out == 0:
                    config.num_operations = 4
                else:
                    config.num_operations = 5

        # Build st_list from config
        st_list = [
            [
                [
                    [x * inner_scale * outer_scale for outer_scale in config.st_list_r]
                    for x in row
                ]
                for row in config.st_list_v
            ]
            for inner_scale in config.st_list_o
        ]

        ready = np.random.uniform(low=0.0, high=config.operation_hour * 60.0, size=num_vehicles)
        vehicle_type = np.random.randint(0, 4, num_vehicles)

        proc_landing = np.zeros((num_vehicles, config.num_pad))
        for i in range(num_vehicles):
            for j in range(config.num_pad):
                proc_landing[i][j] = config.proc_air_o[0] * config.proc_air_v[vehicle_type[i]] * config.proc_air_r[j]

        if config.is_unified_buffer:
            if config.num_buffer > 0:
                proc_buffer = np.zeros((num_vehicles, config.num_buffer))
            else:
                proc_buffer = np.zeros((num_vehicles, 0))
        else:
            if config.num_buffer_in > 0:
                proc_buffer_in = np.zeros((num_vehicles, config.num_buffer_in))
            else:
                proc_buffer_in = np.zeros((num_vehicles, 0))
            if config.num_buffer_out > 0:
                proc_buffer_out = np.zeros((num_vehicles, config.num_buffer_out))
            else:
                proc_buffer_out = np.zeros((num_vehicles, 0))

        proc_gate = np.zeros((num_vehicles, config.num_gate))
        TAT = np.zeros(num_vehicles)
        for i in range(num_vehicles):
            proc_gate[i][0] = config.proc_gate_v[vehicle_type[i]]
        for i in range(num_vehicles):
            for j in range(config.num_gate):
                proc_gate[i][j] = proc_gate[i][0] + 0.15 * (j + 1)
            TAT[i] = np.mean(proc_gate[i])

        proc_takeoff = np.zeros((num_vehicles, config.num_pad))
        for i in range(num_vehicles):
            for j in range(config.num_pad):
                proc_takeoff[i][j] = config.proc_air_o[1] * config.proc_air_v[vehicle_type[i]] * config.proc_air_r[j]

        if config.is_unified_buffer:
            if config.num_buffer == 0:
                proc = [proc_landing, proc_gate, proc_takeoff]
            else:
                proc = [proc_landing, proc_buffer, proc_gate, proc_buffer, proc_takeoff]
        else:
            if config.num_buffer_in == 0:
                if config.num_buffer_out == 0:
                    proc = [proc_landing, proc_gate, proc_takeoff]
                else:
                    proc = [proc_landing, proc_gate, proc_buffer_out, proc_takeoff]
            else:
                if config.num_buffer_out == 0:
                    proc = [proc_landing, proc_buffer_in, proc_gate, proc_takeoff]
                else:
                    proc = [proc_landing, proc_buffer_in, proc_gate, proc_buffer_out, proc_takeoff]

        if config.ETA_ready_diff[0] == config.ETA_ready_diff[1] == config.ETA_ready_diff[2]:
            ETA_ready_diff_arr = config.ETA_ready_diff[0] * np.ones(num_vehicles)
        else:
            ETA_ready_diff_arr = config.ETA_ready_diff[0] + (config.ETA_ready_diff[1] - config.ETA_ready_diff[0])\
                                 * np.random.beta(1.0 + config.lambda_beta_pert * (config.ETA_ready_diff[2] - config.ETA_ready_diff[0]) / (config.ETA_ready_diff[1] - config.ETA_ready_diff[0]),
                                                  1.0 + config.lambda_beta_pert * (config.ETA_ready_diff[1] - config.ETA_ready_diff[2]) / (config.ETA_ready_diff[1] - config.ETA_ready_diff[0]),
                                                  num_vehicles)
        vehicle_planned_arrival_times = ready + ETA_ready_diff_arr
        vehicle_planned_departure_times = vehicle_planned_arrival_times + TAT + config.ETD_margin
        vehicle_planned_gate_close_times = vehicle_planned_arrival_times + TAT + config.gate_close_margin

        o_key_map = {
            0: (0, 0),
            1: (0, config.num_operations - 1),
            2: (config.num_operations - 1, 0),
            3: (config.num_operations - 1, config.num_operations - 1)
        }
        ST = {}
        for o in range(len(st_list)):
            key = o_key_map[o]
            ST[key] = []
            for i in range(num_vehicles):
                row = []
                v_i = vehicle_type[i]
                for j in range(num_vehicles):
                    v_j = vehicle_type[j]
                    row.append(st_list[o][v_i][v_j])
                ST[key].append(row)

        def round_nested_list(lst, digits=2):
            if isinstance(lst, list):
                return [round_nested_list(x, digits) for x in lst]
            elif isinstance(lst, float):
                return round(lst, digits)
            else:
                return lst

        ST_rounded = {k: round_nested_list(v, digits=2) for k, v in ST.items()}

        first_activated_time_of_ready = np.zeros(num_vehicles)
        M = config.operation_hour * 60.0 + (proc_landing.sum() / config.num_pad + proc_gate.sum() / config.num_gate + proc_takeoff.sum() / config.num_pad) / 2

        whether_vehicle_dynamic_arrival = np.isin(vehicle_id, config.dynamic_arrival_v_id)
        vehicle_dynamic_arrival_aware_time = np.full(num_vehicles, -1.0)
        for i in range(num_vehicles):
            if whether_vehicle_dynamic_arrival[i]:
                vehicle_dynamic_arrival_aware_time[i] = np.random.uniform(low=config.dynamic_arrival_aware_time_range[0], high=config.dynamic_arrival_aware_time_range[1])

        return cls(
            config.seed, config.num_operations, config.num_vehicles_per_hour, num_vehicles, vehicle_id, config.num_pad,
            config.num_buffer_in, config.num_gate, config.num_buffer_out, config.num_buffer, num_resource, config.objective_weights,
            config.proc_air_v, config.proc_air_r, config.proc_air_o, config.proc_gate_v, st_list, config.operation_hour * 60.0,
            config.ETA_ready_diff, config.lambda_beta_pert, config.ETD_margin, config.gate_close_margin, config.is_unified_buffer,
            ready, ready, proc, first_activated_time_of_ready, vehicle_planned_arrival_times, vehicle_planned_departure_times,
            vehicle_planned_gate_close_times, ST_rounded, vehicle_type, M, whether_vehicle_dynamic_arrival, vehicle_dynamic_arrival_aware_time,
            proc, config.coe_bridge_from_update_interval
        )


    @classmethod
    def from_scenario_config_true(cls, config: "ScenarioRHCConfig", scenario_exp) -> "Scenario":
        np.random.seed(scenario_exp.seed)

        ready = np.zeros(scenario_exp.num_vehicles)
        for i in range(scenario_exp.num_vehicles):
            if config.disturbance_ready[0] == config.disturbance_ready[1] == config.disturbance_ready[2]:
                ready[i] = scenario_exp.vehicle_arrival_times[i] + config.disturbance_ready[0]
            else:
                if scenario_exp.vehicle_arrival_times[i] <= config.uncertainty_compensation_for_early_ready:
                    disturbance_ready_ = [x * scenario_exp.vehicle_arrival_times[i] / config.uncertainty_compensation_for_early_ready for x in config.disturbance_ready]
                    alpha_ready = 1.0 + config.lambda_beta_pert * (disturbance_ready_[2] - disturbance_ready_[0]) / (disturbance_ready_[1] - disturbance_ready_[0])
                    beta_ready = 1.0 + config.lambda_beta_pert * (disturbance_ready_[1] - disturbance_ready_[2]) / (disturbance_ready_[1] - disturbance_ready_[0])
                    ready[i] = max(scenario_exp.vehicle_arrival_times[i] + disturbance_ready_[0] + (disturbance_ready_[1] - disturbance_ready_[0]) * np.random.beta(alpha_ready, beta_ready), 0.0)
                else:
                    alpha_ready = 1.0 + config.lambda_beta_pert * (config.disturbance_ready[2] - config.disturbance_ready[0]) / (config.disturbance_ready[1] - config.disturbance_ready[0])
                    beta_ready = 1.0 + config.lambda_beta_pert * (config.disturbance_ready[1] - config.disturbance_ready[2]) / (config.disturbance_ready[1] - config.disturbance_ready[0])
                    ready[i] = scenario_exp.vehicle_arrival_times[i] + config.disturbance_ready[0] + (config.disturbance_ready[1] - config.disturbance_ready[0]) * np.random.beta(alpha_ready, beta_ready)

        proc_landing = np.zeros((scenario_exp.num_vehicles, scenario_exp.num_pad))
        proc_gate = np.zeros((scenario_exp.num_vehicles, scenario_exp.num_gate))
        proc_takeoff = np.zeros((scenario_exp.num_vehicles, scenario_exp.num_pad))

        if scenario_exp.is_unified_buffer:
            if scenario_exp.num_buffer > 0:
                proc_buffer = np.zeros((scenario_exp.num_vehicles, config.num_buffer))
            else:
                proc_buffer = np.zeros((scenario_exp.num_vehicles, 0))
        else:
            if scenario_exp.num_buffer_in > 0:
                proc_buffer_in = np.zeros((scenario_exp.num_vehicles, config.num_buffer_in))
            else:
                proc_buffer_in = np.zeros((scenario_exp.num_vehicles, 0))
            if scenario_exp.num_buffer_out > 0:
                proc_buffer_out = np.zeros((scenario_exp.num_vehicles, config.num_buffer_out))
            else:
                proc_buffer_out = np.zeros((scenario_exp.num_vehicles, 0))

        for i in range(scenario_exp.num_vehicles):
            for j in range(scenario_exp.num_pad):
                if config.disturbance_proc[0][0] == config.disturbance_proc[0][1] == config.disturbance_proc[0][2]:
                    proc_landing[i][j] = scenario_exp.proc[0][i][j] + config.disturbance_proc[0][0]
                else:
                    alpha_landing = 1.0 + config.lambda_beta_pert * (config.disturbance_proc[0][2] - config.disturbance_proc[0][0]) / (config.disturbance_proc[0][1] - config.disturbance_proc[0][0])
                    beta_landing = 1.0 + config.lambda_beta_pert * (config.disturbance_proc[0][1] - config.disturbance_proc[0][2]) / (config.disturbance_proc[0][1] - config.disturbance_proc[0][0])
                    proc_landing[i][j] = scenario_exp.proc[0][i][j] + config.disturbance_proc[0][0] + (config.disturbance_proc[0][1] - config.disturbance_proc[0][0]) * np.random.beta(alpha_landing, beta_landing)
                if config.disturbance_proc[2][0] == config.disturbance_proc[2][1] == config.disturbance_proc[2][2]:
                    proc_takeoff[i][j] = scenario_exp.proc[-1][i][j] + config.disturbance_proc[2][0]
                else:
                    alpha_takeoff = 1.0 + config.lambda_beta_pert * (config.disturbance_proc[2][2] - config.disturbance_proc[2][0]) / (config.disturbance_proc[2][1] - config.disturbance_proc[2][0])
                    beta_takeoff = 1.0 + config.lambda_beta_pert * (config.disturbance_proc[2][1] - config.disturbance_proc[2][2]) / (config.disturbance_proc[2][1] - config.disturbance_proc[2][0])
                    proc_takeoff[i][j] = scenario_exp.proc[-1][i][j] + config.disturbance_proc[2][0] + (config.disturbance_proc[2][1] - config.disturbance_proc[2][0]) * np.random.beta(alpha_takeoff, beta_takeoff)

        if config.disturbance_proc[1][0] == config.disturbance_proc[1][1] == config.disturbance_proc[1][2]:
            disturbance_term_gate = config.disturbance_proc[1][0]
        else:
            alpha_gate = 1.0 + config.lambda_beta_pert * (config.disturbance_proc[1][2] - config.disturbance_proc[1][0]) / (config.disturbance_proc[1][1] - config.disturbance_proc[1][0])
            beta_gate = 1.0 + config.lambda_beta_pert * (config.disturbance_proc[1][1] - config.disturbance_proc[1][2]) / (config.disturbance_proc[1][1] - config.disturbance_proc[1][0])
            disturbance_term_gate = config.disturbance_proc[1][0] + (config.disturbance_proc[1][1] - config.disturbance_proc[1][0]) * np.random.beta(alpha_gate, beta_gate)
        if scenario_exp.is_unified_buffer:
            if scenario_exp.num_buffer == 0:
                for i in range(scenario_exp.num_vehicles):
                    for j in range(scenario_exp.num_gate):
                        proc_gate[i][j] = scenario_exp.proc[1][i][j] + disturbance_term_gate
                proc = [proc_landing, proc_gate, proc_takeoff]
            else:
                for i in range(scenario_exp.num_vehicles):
                    for j in range(scenario_exp.num_gate):
                        proc_gate[i][j] = scenario_exp.proc[2][i][j] + disturbance_term_gate
                proc = [proc_landing, proc_buffer, proc_gate, proc_buffer, proc_takeoff]
        else:
            if scenario_exp.num_buffer_in == 0:
                for i in range(scenario_exp.num_vehicles):
                    for j in range(scenario_exp.num_gate):
                        proc_gate[i][j] = scenario_exp.proc[1][i][j] + disturbance_term_gate
                if scenario_exp.num_buffer_out == 0:
                    proc = [proc_landing, proc_gate, proc_takeoff]
                else:
                    proc = [proc_landing, proc_gate, proc_buffer_out, proc_takeoff]
            else:
                for i in range(scenario_exp.num_vehicles):
                    for j in range(scenario_exp.num_gate):
                        proc_gate[i][j] = scenario_exp.proc[2][i][j] + disturbance_term_gate
                if scenario_exp.num_buffer_out == 0:
                    proc = [proc_landing, proc_buffer_in, proc_gate, proc_takeoff]
                else:
                    proc = [proc_landing, proc_buffer_in, proc_gate, proc_buffer_out, proc_takeoff]

        proc_real = copy.deepcopy(proc)
        for i in range(len(config.dynamic_proc_v_id)):
            proc_real[config.dynamic_proc_op[i]][config.dynamic_proc_v_id[i]] += config.dynamic_proc_inc_time[i]

        return cls(
            scenario_exp.seed, scenario_exp.num_operations, scenario_exp.num_vehicles_per_hour, scenario_exp.num_vehicles,
            scenario_exp.vehicle_id, scenario_exp.num_pad, scenario_exp.num_buffer_in, scenario_exp.num_gate, scenario_exp.num_buffer_out,
            scenario_exp.num_buffer, scenario_exp.num_resource, scenario_exp.objective_weights, scenario_exp.proc_air_v,
            scenario_exp.proc_air_r, scenario_exp.proc_air_o, scenario_exp.proc_gate_v, scenario_exp.st_list, config.operation_hour * 60.0,
            scenario_exp.ETA_ready_diff, scenario_exp.lambda_beta_pert, scenario_exp.ETD_margin, scenario_exp.gate_close_margin,
            scenario_exp.is_unified_buffer, ready, ready, proc, scenario_exp.first_activated_time_of_ready,
            scenario_exp.vehicle_planned_arrival_times, scenario_exp.vehicle_planned_departure_times, scenario_exp.vehicle_planned_gate_close_times,
            scenario_exp.ST, scenario_exp.vehicle_type, scenario_exp.big_M, scenario_exp.whether_vehicle_dynamic_arrival,
            scenario_exp.vehicle_dynamic_arrival_aware_time, proc_real, scenario_exp.coe_bridge_from_update_interval
        )


    @classmethod
    def scenario_to_instance_exp(cls, scenario_exp, scenario_true, instance_exp_pre, activated_vehicle_id_exp_pre,
                                 scheduling_horizon, update_interval, processing_vehicles_id, processing_vehicles_op,
                                 remaining_proc_time):

        dynamic_vehicle_arrival_times = copy.deepcopy(scenario_exp.vehicle_arrival_times)
        for i in range(len(activated_vehicle_id_exp_pre)):
            if dynamic_vehicle_arrival_times[activated_vehicle_id_exp_pre[i]] != instance_exp_pre.vehicle_arrival_times[i]:
                if instance_exp_pre.vehicle_arrival_times[i] <= scheduling_horizon[0]:
                    if activated_vehicle_id_exp_pre[i] not in processing_vehicles_id and instance_exp_pre.vehicle_arrival_times[i] >= scheduling_horizon[0] - update_interval:
                        dynamic_vehicle_arrival_times[activated_vehicle_id_exp_pre[i]] = scheduling_horizon[0]
                    else:
                        pass
                else:
                    dynamic_vehicle_arrival_times[activated_vehicle_id_exp_pre[i]] = instance_exp_pre.vehicle_arrival_times[i]

        ready_in_horizon_vehicle_id_exp = [i for i, x in enumerate(dynamic_vehicle_arrival_times) if
                                           scheduling_horizon[0] <= x <= scheduling_horizon[1] *
                                           (1.0 - scenario_exp.whether_vehicle_dynamic_arrival[i]) + (scheduling_horizon[0] +
                                           scenario_exp.vehicle_dynamic_arrival_aware_time[i]) * scenario_exp.whether_vehicle_dynamic_arrival[i]]

        newly_ready_in_horizon_vehicle_id_exp = [i for i, x in enumerate(dynamic_vehicle_arrival_times) if
                                                 scheduling_horizon[1] * (1.0 - scenario_exp.whether_vehicle_dynamic_arrival[i]) +
                                                 (scheduling_horizon[0] + scenario_exp.vehicle_dynamic_arrival_aware_time[i]) *
                                                 scenario_exp.whether_vehicle_dynamic_arrival[i] - update_interval <= x <= scheduling_horizon[1] *
                                                 (1.0 - scenario_exp.whether_vehicle_dynamic_arrival[i]) +
                                                 (scheduling_horizon[0] + scenario_exp.vehicle_dynamic_arrival_aware_time[i]) *
                                                 scenario_exp.whether_vehicle_dynamic_arrival[i]]

        ready_in_horizon_vehicle_id_exp = [
            vid for vid in ready_in_horizon_vehicle_id_exp
            if vid not in processing_vehicles_id
        ]
        activated_vehicle_id_exp = processing_vehicles_id + ready_in_horizon_vehicle_id_exp

        proc_in_horizon = [
            [[row_res for row_res in scenario_exp.proc[op][v]] for v in activated_vehicle_id_exp]
            for op in range(len(scenario_exp.proc))
        ]
        for i in range(len(processing_vehicles_id)):
            for j in range(processing_vehicles_op[i] + 1):
                for k in range(len(proc_in_horizon[j][i])):
                    if j == processing_vehicles_op[i]:
                        if proc_in_horizon[j][i][k] > 0:
                            proc_in_horizon[j][i][k] = remaining_proc_time[i]
                    else:
                        proc_in_horizon[j][i][k] = 0.0

        ready_in_horizon_exp = scenario_exp.vehicle_arrival_times[activated_vehicle_id_exp]
        ready_in_horizon_true = scenario_true.vehicle_arrival_times[activated_vehicle_id_exp]

        for i in range(len(processing_vehicles_id)):
            ready_in_horizon_exp[i] = 0.0

        for i in range(len(processing_vehicles_id), len(activated_vehicle_id_exp)):
            if activated_vehicle_id_exp[i] not in newly_ready_in_horizon_vehicle_id_exp:
                half_width_for_stochastic_bridge = update_interval * scenario_exp.coe_bridge_from_update_interval\
                                              * (ready_in_horizon_true[i] - scheduling_horizon[0])\
                                              / (ready_in_horizon_true[i] - scenario_exp.first_activated_time_of_ready[activated_vehicle_id_exp[i]])
                if half_width_for_stochastic_bridge >= 0.0:
                    ready_in_horizon_exp[i] = max(scenario_exp.vehicle_arrival_times_init[activated_vehicle_id_exp[i]] \
                                              + (scheduling_horizon[0] - scenario_exp.first_activated_time_of_ready[activated_vehicle_id_exp[i]]) \
                                              * (ready_in_horizon_true[i] - scenario_exp.vehicle_arrival_times_init[activated_vehicle_id_exp[i]]) \
                                              / (ready_in_horizon_true[i] - scenario_exp.first_activated_time_of_ready[activated_vehicle_id_exp[i]]) \
                                              + scenario_exp.bridge_rng.uniform(-half_width_for_stochastic_bridge, half_width_for_stochastic_bridge), scheduling_horizon[0])
            else:
                if scheduling_horizon[0] == 0.0 and ready_in_horizon_exp[i] < scheduling_horizon[1] - update_interval:
                    scenario_exp.first_activated_time_of_ready[activated_vehicle_id_exp[i]] = ready_in_horizon_exp[i] - scheduling_horizon[1]
                else:
                    scenario_exp.first_activated_time_of_ready[activated_vehicle_id_exp[i]] = scheduling_horizon[0]

        vehicle_planned_arrival_times_in_horizon = scenario_exp.vehicle_planned_arrival_times[activated_vehicle_id_exp]
        vehicle_planned_departure_times_in_horizon = scenario_exp.vehicle_planned_departure_times[activated_vehicle_id_exp]
        vehicle_planned_gate_close_times_in_horizon = scenario_exp.vehicle_planned_gate_close_times[activated_vehicle_id_exp]
        vehicle_type_in_horizon = scenario_exp.vehicle_type[activated_vehicle_id_exp]

        o_key_map = {
            0: (0, 0),
            1: (0, scenario_exp.num_operations - 1),
            2: (scenario_exp.num_operations - 1, 0),
            3: (scenario_exp.num_operations - 1, scenario_exp.num_operations - 1)
        }
        ST_in_horizon = {}
        for o in range(len(scenario_exp.st_list)):
            key = o_key_map[o]
            ST_in_horizon[key] = []
            for i in activated_vehicle_id_exp:
                row = []
                v_i = scenario_exp.vehicle_type[i]
                for j in activated_vehicle_id_exp:
                    v_j = scenario_exp.vehicle_type[j]
                    row.append(scenario_exp.st_list[o][v_i][v_j])
                ST_in_horizon[key].append(row)

        def round_nested_list(lst, digits=2):
            if isinstance(lst, list):
                return [round_nested_list(x, digits) for x in lst]
            elif isinstance(lst, float):
                return round(lst, digits)
            else:
                return lst

        ST_rounded_in_horizon = {k: round_nested_list(v, digits=2) for k, v in ST_in_horizon.items()}

        M = (scheduling_horizon[1] - scheduling_horizon[0]) + 100.0

        return Instance(
            scenario_exp.seed, scenario_exp.num_operations, len(activated_vehicle_id_exp), scenario_exp.num_pad, scenario_exp.num_buffer_in,
            scenario_exp.num_gate, scenario_exp.num_buffer_out, scenario_exp.num_buffer, scenario_exp.num_resource, scenario_exp.objective_weights,
            scenario_exp.proc_air_v, scenario_exp.proc_air_r, scenario_exp.proc_air_o, scenario_exp.proc_gate_v, scenario_exp.st_list,
            scenario_exp.maximum_arrival_time, scenario_exp.ETA_ready_diff, scenario_exp.lambda_beta_pert, scenario_exp.ETD_margin,
            scenario_exp.gate_close_margin, scenario_exp.is_unified_buffer, ready_in_horizon_exp, proc_in_horizon,
            vehicle_planned_arrival_times_in_horizon, vehicle_planned_departure_times_in_horizon, vehicle_planned_gate_close_times_in_horizon,
            ST_rounded_in_horizon, vehicle_type_in_horizon, M
        ), activated_vehicle_id_exp, ready_in_horizon_vehicle_id_exp

    @classmethod
    def scenario_to_instance_true(cls, scenario_true, current_time):
        ready_in_horizon_vehicle_id_true = [i for i, x in enumerate(scenario_true.vehicle_arrival_times) if
                                            0.0 <= x <= current_time]
        activated_vehicle_id_true = ready_in_horizon_vehicle_id_true

        proc_in_horizon = [
            [[row_res for row_res in scenario_true.proc_real[op][v]] for v in activated_vehicle_id_true]
            for op in range(len(scenario_true.proc_real))
        ]

        ready_in_horizon = scenario_true.vehicle_arrival_times[activated_vehicle_id_true]

        vehicle_planned_arrival_times_in_horizon = scenario_true.vehicle_planned_arrival_times[activated_vehicle_id_true]
        vehicle_planned_departure_times_in_horizon = scenario_true.vehicle_planned_departure_times[activated_vehicle_id_true]
        vehicle_planned_gate_close_times_in_horizon = scenario_true.vehicle_planned_gate_close_times[activated_vehicle_id_true]
        vehicle_type_in_horizon = scenario_true.vehicle_type[activated_vehicle_id_true]

        o_key_map = {
            0: (0, 0),
            1: (0, scenario_true.num_operations - 1),
            2: (scenario_true.num_operations - 1, 0),
            3: (scenario_true.num_operations - 1, scenario_true.num_operations - 1)
        }
        ST_in_horizon = {}
        for o in range(len(scenario_true.st_list)):
            key = o_key_map[o]
            ST_in_horizon[key] = []
            for i in activated_vehicle_id_true:
                row = []
                v_i = scenario_true.vehicle_type[i]
                for j in activated_vehicle_id_true:
                    v_j = scenario_true.vehicle_type[j]
                    row.append(scenario_true.st_list[o][v_i][v_j])
                ST_in_horizon[key].append(row)

        def round_nested_list(lst, digits=2):
            if isinstance(lst, list):
                return [round_nested_list(x, digits) for x in lst]
            elif isinstance(lst, float):
                return round(lst, digits)
            else:
                return lst

        ST_rounded_in_horizon = {k: round_nested_list(v, digits=2) for k, v in ST_in_horizon.items()}

        return Instance(
            scenario_true.seed, scenario_true.num_operations, len(activated_vehicle_id_true), scenario_true.num_pad, scenario_true.num_buffer_in,
            scenario_true.num_gate, scenario_true.num_buffer_out, scenario_true.num_buffer, scenario_true.num_resource, scenario_true.objective_weights,
            scenario_true.proc_air_v, scenario_true.proc_air_r, scenario_true.proc_air_o, scenario_true.proc_gate_v, scenario_true.st_list,
            scenario_true.maximum_arrival_time, scenario_true.ETA_ready_diff, scenario_true.lambda_beta_pert, scenario_true.ETD_margin,
            scenario_true.gate_close_margin, scenario_true.is_unified_buffer, ready_in_horizon, proc_in_horizon,
            vehicle_planned_arrival_times_in_horizon, vehicle_planned_departure_times_in_horizon, vehicle_planned_gate_close_times_in_horizon,
            ST_rounded_in_horizon, vehicle_type_in_horizon, scenario_true.big_M
        ), activated_vehicle_id_true
