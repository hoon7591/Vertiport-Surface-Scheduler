from typing import Any, List
import numpy as np
from dataclasses import dataclass, field


"""
Hyper Parameter Lists for Problem Generation in InstanceConfig
# seed: for fixing random seed
# num_ops: number of operations in the vertiport service (depending on the existence of buffer_in and buffer_out, the num_ops can be 3, 4, or 5)
# num_vehicle: number of UAMs supposed to use vertiport
# num_pad: number of pads for landing and take-off in vertiport
# num_buffer_in: capacity of waiting space before occupying a gate
# num_gate: number of gate/parking(charging) slot in vertiport
# num_buffer_out: capacity of waiting space before take-off
# num_buffer: capacity of waiting space in unified buffer case
# num_resource: number of all resources in vertiport
# weights: weights in objective function (1D list, len = 2)
# proc_air_v: processing time of each vehicle type for landing and take-off operations (1D list, len = 4)
# proc_air_r: coefficient of processing time for landing and take-off operations depending on pads assignment (1D list, len = num_pad)
# proc_air_o: coefficient reflecting variation of processing time between landing and take-off operations (1D list, [landing, take-off])
# proc_gate_v: processing time of each vehicle type for gate operation (1D list, len = 4)
# st_list_v: separation time list for each vehicle combinations (2D list, dim = 4 by 4)
#            types of vehicle => light/copter; type = 0, light/fixed-wing; type = 1, heavy/copter; type = 2, heavy/fixed-wing; type = 3
#            (left most one is most vulnerable one to endure turbulence, and the farther to the right, the more resistant to turbulence)
#            ex) st_list_v[3][0] => heavy/fixed-wing UAM precedes light/copter UAM
# st_list_o: coefficient depending on operations pair for separation time setting (1D list, len = 4)
#            (o, o') => [coe of (1, 1), coe of (1, 5), coe of (5, 1), coe of (5, 5)]
# st_list_r: coefficient depending on resource for separation time setting (1D list, len = num_pad)
#            len(st_list_r) == num_pad
# st_list: concatenated separation list (4D list, automatically generated from st_list_v, st_list_o, and st_list_r)
#          st_list[operation_pair, type of v, type of v', resource]
# ready_max: maximum of ready time
# ETA_ready_diff: ETA(=due_a) - ready for all vehicles (1D list)
#                 [nominal, width] (width/2 corresponds to 1-sigma deviation of normal distribution; 68.27% of values are within this range from nominal value)
# ETD_margin: ETD(=due_d) = ETA + TAT + "ETD_margin"
# gate_close_margin: ETA + TAT + "gate_close_margin" (for no-early-departure const.)
# unified_buffer: If True, buffer_in and buffer_out are unified into a single buffer

# Info. of generate_instance function
# ready: ready[vehicle]
# proc: proc[operation][vehicle][resource]
# due_a: due_a[vehicle]
# due_d: due_d[vehicle]
# vehicle_type: vehicle_type[vehicle]
# ST: ST[operation_pair, v, v', resource(pad)] (dictionary; keys = operation_pair tuple such as (1, 1), (5, 1), ...)
# weights: weights in objective function (1D list)
#          => weights for obj definition is already declared in hyper_param_setting
# M: big-M in formulation

# Notes:
# 1. Time unit is minute
# 2. If you change 'num_pad', you need to also change 'st_list_r' => len(st_list_r) = num_pad
"""


@dataclass
class InstanceConfig:
    """Configuration class for generating an instance of the UAM-VS-Scheduling Problem.
    This class holds the parameters required to create a problem instance, including
    the number of operations, vehicles, pads, gates, and various processing times.
    """
    seed: int = 42
    num_operations: int = 5
    num_vehicles: int = 20
    num_pad: int = 2
    num_buffer_in: Any = None
    num_gate: int = 10
    num_buffer_out: Any = None # not used for is_unified_buffer = True: if True, this is merged with buffer_in #TODO : revise implicitly
    num_buffer: Any = None # only used for is_unified_buffer = True
    objective_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])
    proc_air_v: List[float] = field(default_factory=lambda: [2.0, 2.2, 2.8, 3.0])
    proc_air_r: List[float] = field(default_factory=lambda: [0.8, 0.9, 1.0, 1.0])
    proc_air_o: List[float] = field(default_factory=lambda: [1.2, 1.0])
    proc_gate_v: List[int] = field(default_factory=lambda: [15, 17, 23, 25])
    st_list_v: Any = field(default_factory=lambda: [
        [1.25, 1.0, 1.0, 1.0],
        [1.5, 1.25, 1.0, 1.0],
        [1.75, 1.5, 1.25, 1.0],
        [2.0, 1.75, 1.5, 1.25]
    ])
    st_list_o: List[float] = field(default_factory=lambda: [0.8, 0.6, 0.5, 1.0])
    st_list_r: List[float] = field(default_factory=lambda: [1.0, 0.7, 0.8, 1.0])
    ready_max: float = 100.0
    ETA_ready_diff: List[float] = field(default_factory=lambda: [3, 3])
    ETD_margin: float = 5.0
    gate_close_margin: float = 3.0
    is_unified_buffer: bool = False


class Instance:
    def __init__(
        self,
        seed: int,
        num_operations: int,
        num_vehicles: int,
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
        maximum_arrival_time: float, # defines instance's horizon 
        ETA_ready_diff: List[float],
        ETD_margin: float,
        gate_close_margin: float,
        is_unified_buffer: bool,
        vehicle_arrival_times: np.ndarray,
        proc: list,
        vehicle_planned_arrival_times: np.ndarray,
        vehicle_planned_departure_times: np.ndarray,
        vehicle_planned_gate_close_times: np.ndarray,
        ST: dict,
        vehicle_type: np.ndarray,
        M: float,
    ):
        self.seed = seed
        self.num_operations = num_operations
        self.num_vehicles = num_vehicles
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
        self.ETD_margin = ETD_margin # TODO : check if it needs?
        self.gate_close_margin = gate_close_margin
        self.is_unified_buffer = is_unified_buffer
        self.vehicle_arrival_times = vehicle_arrival_times
        self.proc = proc
        self.vehicle_planned_arrival_times = vehicle_planned_arrival_times
        self.vehicle_planned_departure_times = vehicle_planned_departure_times
        self.vehicle_planned_gate_close_times = vehicle_planned_gate_close_times
        self.ST = ST
        self.vehicle_type = vehicle_type
        self.big_M = M

    @classmethod
    def from_config(cls, config: "InstanceConfig") -> "Instance":
        np.random.seed(config.seed)
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

        ready = np.random.uniform(low=0.0, high=config.ready_max, size=config.num_vehicles)
        vehicle_type = np.random.randint(0, 4, config.num_vehicles)

        proc_landing = np.zeros((config.num_vehicles, config.num_pad))
        for i in range(config.num_vehicles):
            for j in range(config.num_pad):
                proc_landing[i][j] = config.proc_air_o[0] * config.proc_air_v[vehicle_type[i]] * config.proc_air_r[j]

        if config.is_unified_buffer:
            if config.num_buffer > 0:
                proc_buffer = np.zeros((config.num_vehicles, config.num_buffer))
            else:
                proc_buffer = np.zeros((config.num_vehicles, 0))
        else:
            if config.num_buffer_in > 0:
                proc_buffer_in = np.zeros((config.num_vehicles, config.num_buffer_in))
            else:
                proc_buffer_in = np.zeros((config.num_vehicles, 0))
            if config.num_buffer_out > 0:
                proc_buffer_out = np.zeros((config.num_vehicles, config.num_buffer_out))
            else:
                proc_buffer_out = np.zeros((config.num_vehicles, 0))

        proc_gate = np.zeros((config.num_vehicles, config.num_gate))
        TAT = np.zeros(config.num_vehicles)
        for i in range(config.num_vehicles):
            proc_gate[i][0] = config.proc_gate_v[vehicle_type[i]]
            TAT[i] = proc_gate[i][0]
        for i in range(config.num_vehicles):
            for j in range(config.num_gate):
                proc_gate[i][j] = proc_gate[i][0] + 0.15 * (j + 1)

        proc_takeoff = np.zeros((config.num_vehicles, config.num_pad))
        for i in range(config.num_vehicles):
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

        ETA_ready_diff_arr = config.ETA_ready_diff[0] + config.ETA_ready_diff[1] / 2 * np.random.randn(config.num_vehicles)
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
            for i in range(config.num_vehicles):
                row = []
                v_i = vehicle_type[i]
                for j in range(config.num_vehicles):
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

        M = config.ready_max + (proc_landing.sum() / config.num_pad + proc_gate.sum() / config.num_gate + proc_takeoff.sum() / config.num_pad) / 2

        return cls(
            config.seed, config.num_operations, config.num_vehicles, config.num_pad, config.num_buffer_in, config.num_gate,
            config.num_buffer_out, config.num_buffer, num_resource, config.objective_weights, config.proc_air_v, config.proc_air_r,
            config.proc_air_o, config.proc_gate_v, st_list, config.ready_max, config.ETA_ready_diff, config.ETD_margin,
            config.gate_close_margin, config.is_unified_buffer, ready, proc, vehicle_planned_arrival_times,
            vehicle_planned_departure_times, vehicle_planned_gate_close_times, ST_rounded, vehicle_type, M
        )
