from typing import Any, List
import numpy as np
from dataclasses import dataclass, field

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
    proc_nominal: List[float] = field(default_factory=lambda: [3, 0, 20, 0, 3])
    proc_width: List[float] = field(default_factory=lambda: [2, 0, 10, 0, 2])
    st_list_v: Any = field(default_factory=lambda: [
        [1.5, 1, 1, 1],
        [2, 1.5, 1, 1],
        [2.5, 2, 1.5, 1],
        [3, 2.5, 1.5, 1.5]
    ])
    st_list_o: List[float] = field(default_factory=lambda: [0.8, 0.6, 0.5, 1.0])
    st_list_r: List[float] = field(default_factory=lambda: [1.0, 0.7])
    ready_max: float = 100.0
    ETA_ready_diff: List[float] = field(default_factory=lambda: [2, 10])
    ETD_margin: float = 5.0


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
        proc_nominal: List[float],
        proc_width: List[float],
        st_list: Any,
        ready_max: float,
        ETA_ready_diff: List[float],
        ETD_margin: float,
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
        self.proc_nominal = proc_nominal
        self.proc_width = proc_width
        self.st_list = st_list
        self.ready_max = ready_max
        self.ETA_ready_diff = ETA_ready_diff
        self.ETD_margin = ETD_margin
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
        proc_landing = np.random.uniform(
            low=config.proc_nominal[0] - config.proc_width[0] / 2,
            high=config.proc_nominal[0] + config.proc_width[0] / 2,
            size=(config.num_vehicle, config.num_pad)
        )
        proc_buffer_in = np.zeros((config.num_vehicle, config.num_buffer_in))
        proc_gate = np.zeros((config.num_vehicle, config.num_gate))
        TAT = np.zeros(config.num_vehicle)
        for i in range(config.num_vehicle):
            proc_gate[i][0] = np.random.uniform(
                low=config.proc_nominal[2] - config.proc_width[2] / 2,
                high=config.proc_nominal[2] + config.proc_width[2] / 2
            )
            TAT[i] = proc_gate[i][0]
        for i in range(config.num_vehicle):
            for j in range(config.num_gate):
                proc_gate[i][j] = proc_gate[i][0] + 0.15 * (j + 1)
        proc_buffer_out = np.zeros((config.num_vehicle, config.num_buffer_out))
        proc_takeoff = np.random.uniform(
            low=config.proc_nominal[4] - config.proc_width[4] / 2,
            high=config.proc_nominal[4] + config.proc_width[4] / 2,
            size=(config.num_vehicle, config.num_pad)
        )
        proc = [proc_landing, proc_buffer_in, proc_gate, proc_buffer_out, proc_takeoff]

        ETA_ready_diff_arr = config.ETA_ready_diff[0] + config.ETA_ready_diff[0] / 2 * np.random.randn(config.num_vehicle)
        due_a = ready + ETA_ready_diff_arr
        due_d = due_a + TAT + config.ETD_margin
        vehicle_type = np.random.randint(0, 4, config.num_vehicle)

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

        M = config.ready_max + proc_landing.sum() / config.num_pad + proc_gate.sum() / config.num_gate + proc_takeoff.sum() / config.num_pad

        return cls(
            config.seed, config.num_ops, config.num_vehicle, config.num_pad, config.num_buffer_in, config.num_gate, config.num_buffer_out,
            num_resource, config.weights, config.proc_nominal, config.proc_width, st_list, config.ready_max, config.ETA_ready_diff, config.ETD_margin,
            ready, proc, due_a, due_d, ST_rounded, vehicle_type, M
        )