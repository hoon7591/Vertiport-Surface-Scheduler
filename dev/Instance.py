from typing import Any, List
import numpy as np
from dataclasses import dataclass, field


"""
Hyper Parameter Lists for Problem Generation in InstanceConfig
seed: for fixing random seed
num_ops: number of operations in the vertiport service
num_vehicle: number of UAMs supposed to use vertiport
num_pad: number of pads for landing and take-off in vertiport
num_buffer_in: capacity of waiting space before occupying a gate
num_gate: number of gate/parking(charging) slot in vertiport
num_buffer_out: capacity of waiting space before take-off
num_resource: number of all resources in vertiport
weights: weights in objective function (1D list, len = 2)
proc_air_v: processing time of each vehicle type for landing and take-off operations (1D list, len = 4)
proc_air_r: coefficient of processing time for landing and take-off operations depending on pads assignment (1D list, len = num_pad)
proc_air_o: coefficient reflecting variation of processing time between landing and take-off operations (1D list, [landing, take-off])
proc_gate_v: processing time of each vehicle type for gate operation (1D list, len = 4)
st_list_v: separation time list for each vehicle combinations (2D list, dim = 4 by 4)
           types of vehicle => light/copter; type = 0, light/fixed-wing; type = 1, heavy/copter; type = 2, heavy/fixed-wing; type = 3
           (left most one is most vulnerable one to endure turbulence, and the farther to the right, the more resistant to turbulence)
           ex) st_list_v[3][0] => heavy/fixed-wing UAM precedes light/copter UAM
st_list_o: coefficient depending on operations pair for separation time setting (1D list, len = 4)
           (o, o') => [coe of (1, 1), coe of (1, 5), coe of (5, 1), coe of (5, 5)]
st_list_r: coefficient depending on resource for separation time setting (1D list, len = num_pad)
           len(st_list_r) == num_pad
st_list: concatenated separation list (4D list, automatically generated from st_list_v, st_list_o, and st_list_r)
         st_list[operation_pair, type of v, type of v', resource]
ready_max: maximum of ready time
ETA_ready_diff: ETA(=due_a) - ready for all vehicles (1D list)
                [nominal, width] (width/2 corresponds to 2-sigma deviation of normal distribution; 95.45% of values are within this range from nominal value)
ETD_margin: ETD(=due_d) = ETA + TAT + "ETD_margin"
unified_buffer: If True, buffer_in and buffer_out are unified into a single buffer
"""


"""
Info. of generate_instance function
ready: ready[vehicle]
proc: proc[operation][vehicle][resource]
due_a: due_a[vehicle]
due_d: due_d[vehicle]
vehicle_type: vehicle_type[vehicle]
ST: ST[operation_pair, v, v', resource(pad)] (dictionary; keys = operation_pair tuple such as (1, 1), (5, 1), ...)
weights: weights in objective function (1D list)
         => weights for obj definition is already declared in hyper_param_setting
M: big-M in formulation
"""


"""
1. Time unit is minute
2. If you change 'num_pad', you need to also change 'st_list_r' => len(st_list_r) = num_pad
"""


@dataclass
class InstanceConfig:
    seed: int = 42
    num_ops: int = 5
    num_vehicle: int = 20
    num_pad: int = 2
    num_buffer_in: int = 3
    num_gate: int = 10
    num_buffer_out: int = 3
    weights: List[float] = field(default_factory=lambda: [0.5, 0.5])
    proc_air_v: List[float] = field(default_factory=lambda: [2.0, 2.2, 2.8, 3.0])
    proc_air_r: List[float] = field(default_factory=lambda: [1.0, 0.8])
    proc_air_o: List[float] = field(default_factory=lambda: [1.2, 1.0])
    proc_gate_v: List[int] = field(default_factory=lambda: [15, 17, 23, 25])
    st_list_v: Any = field(default_factory=lambda: [
        [1.5, 1, 1, 1],
        [2, 1.5, 1, 1],
        [2.5, 2, 1.5, 1],
        [3, 2.5, 1.5, 1.5]
    ])
    st_list_o: List[float] = field(default_factory=lambda: [0.8, 0.6, 0.5, 1.0])
    st_list_r: List[float] = field(default_factory=lambda: [1.0, 0.7])
    ready_max: float = 100.0
    ETA_ready_diff: List[float] = field(default_factory=lambda: [3, 10])
    ETD_margin: float = 5.0
    unified_buffer: bool = False


class Instance:
    def __init__(
        self,
        seed: int,
        num_ops: int,
        num_vehicle: int,
        num_pad: int,
        num_buffer_in: int,
        num_gate: int,
        num_buffer_out: int,
        num_resource: int,
        weights: List[float],
        proc_air_v: List[float],
        proc_air_r: List[float],
        proc_air_o: List[float],
        proc_gate_v: List[int],
        st_list: Any,
        ready_max: float,
        ETA_ready_diff: List[float],
        ETD_margin: float,
        unified_buffer: bool,
        ready: np.ndarray,
        proc: list,
        due_a: np.ndarray,
        due_d: np.ndarray,
        ST: dict,
        vehicle_type: np.ndarray,
        M: float,
    ):
        self.seed = seed
        self.num_ops = num_ops
        self.num_vehicle = num_vehicle
        self.num_pad = num_pad
        self.num_buffer_in = num_buffer_in
        self.num_gate = num_gate
        self.num_buffer_out = num_buffer_out
        self.num_resource = num_resource
        self.weights = weights
        self.proc_air_v = proc_air_v
        self.proc_air_r = proc_air_r
        self.proc_air_o = proc_air_o
        self.proc_gate_v = proc_gate_v
        self.st_list = st_list
        self.ready_max = ready_max
        self.ETA_ready_diff = ETA_ready_diff
        self.ETD_margin = ETD_margin
        self.unified_buffer = unified_buffer
        self.ready = ready
        self.proc = proc
        self.due_a = due_a
        self.due_d = due_d
        self.ST = ST
        self.vehicle_type = vehicle_type
        self.M = M

    @classmethod
    def from_config(cls, config: "InstanceConfig") -> "Instance":
        np.random.seed(config.seed)
        if config.unified_buffer:
            num_resource = config.num_pad + config.num_buffer_in + config.num_gate
        else:
            num_resource = config.num_pad + config.num_buffer_in + config.num_gate + config.num_buffer_out

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

        ready = np.random.uniform(low=0.0, high=config.ready_max, size=config.num_vehicle)
        vehicle_type = np.random.randint(0, 4, config.num_vehicle)

        proc_landing = np.zeros((config.num_vehicle, config.num_pad))
        for i in range(config.num_vehicle):
            for j in range(config.num_pad):
                proc_landing[i][j] = config.proc_air_o[0] * config.proc_air_v[vehicle_type[i]] * config.proc_air_r[j]

        proc_buffer_in = np.zeros((config.num_vehicle, config.num_buffer_in))
        proc_gate = np.zeros((config.num_vehicle, config.num_gate))
        TAT = np.zeros(config.num_vehicle)
        for i in range(config.num_vehicle):
            proc_gate[i][0] = config.proc_gate_v[vehicle_type[i]]
            TAT[i] = proc_gate[i][0]
        for i in range(config.num_vehicle):
            for j in range(config.num_gate):
                proc_gate[i][j] = proc_gate[i][0] + 0.15 * (j + 1)

        proc_buffer_out = np.zeros((config.num_vehicle, config.num_buffer_out))
        proc_takeoff = np.zeros((config.num_vehicle, config.num_pad))
        for i in range(config.num_vehicle):
            for j in range(config.num_pad):
                proc_takeoff[i][j] = config.proc_air_o[1] * config.proc_air_v[vehicle_type[i]] * config.proc_air_r[j]

        proc = [proc_landing, proc_buffer_in, proc_gate, proc_buffer_out, proc_takeoff]

        ETA_ready_diff_arr = config.ETA_ready_diff[0] + config.ETA_ready_diff[0] / 4 * np.random.randn(config.num_vehicle)
        due_a = ready + ETA_ready_diff_arr
        due_d = due_a + TAT + config.ETD_margin

        o_key_map = {
            0: (0, 0),
            1: (0, config.num_ops - 1),
            2: (config.num_ops - 1, 0),
            3: (config.num_ops - 1, config.num_ops - 1)
        }
        ST = {}
        for o in range(len(st_list)):
            key = o_key_map[o]
            ST[key] = []
            for i in range(config.num_vehicle):
                row = []
                v_i = vehicle_type[i]
                for j in range(config.num_vehicle):
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
            config.seed, config.num_ops, config.num_vehicle, config.num_pad, config.num_buffer_in, config.num_gate, config.num_buffer_out,
            num_resource, config.weights, config.proc_air_v, config.proc_air_r, config.proc_air_o, config.proc_gate_v,
            st_list, config.ready_max, config.ETA_ready_diff, config.ETD_margin, config.unified_buffer,
            ready, proc, due_a, due_d, ST_rounded, vehicle_type, M
        )
