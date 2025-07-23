import numpy as np
from Instance import Instance, InstanceConfig


"""
Hyper Parameters List for Problem Generation
seed: for fixing random seed
num_ops: number of operations in the vertiport service
num_vehicle: number of UAMs supposed to use vertiport
num_pad: number of pads for landing and take-off in vertiport
num_buffer_in: capacity of waiting space before occupying a gate
num_gate: number of gate/parking(charging) slot in vertiport
num_buffer_out: capacity of waiting space before take-off
num_resource: number of all resources in vertiport
weights: weights in objective function (1D list)
proc_nominal: nominal proc. time for each operation (1D list)
proc_width: width of interval of proc. time distribution for each operation (1D list)
st_list_v: separation time list for each vehicle combinations (2D list)
           types of vehicle => light/copter; type = 0, light/fixed-wing; type = 1, heavy/copter; type = 2, heavy/fixed-wing; type = 3
           (left most one is most vulnerable one to endure turbulence, and the farther to the right, the more resistant to turbulence)
           ex) st_list_v[3][0] => heavy/fixed-wing UAM precedes light/copter UAM
st_list_o: coefficient depending on operations pair for separation time setting (1D list)
           (o, o') => [coe of (1, 1), coe of (1, 5), coe of (5, 1), coe of (5, 5)]
st_list_r: coefficient depending on resource for separation time setting (1D list)
           len(st_list_r) == num_pad
st_list: concatenated separation list (4D list)
         st_list[operation_pair, type of v, type of v', resource]
ready_max: maximum of ready time
ETA_ready_diff: ETA(=due_a) - ready for all vehicles (1D list)
                [nominal, width]
ETD_margin: ETD(=due_d) = ETA + TAT + "ETD_margin"

time unit is minute 
"""


def hyper_param_setting():
    seed = 42
    num_ops = 5
    num_vehicle = 20
    num_pad = 2
    num_buffer_in = 3
    num_gate = 10
    num_buffer_out = 3
    num_resource = num_pad + num_buffer_in + num_gate + num_buffer_out
    weights = [0.5, 0.5]        # alpha_a, alpha_d
    proc_nominal = [3, 0, 20, 0, 3]         # landing, buffer-in, gate, buffer-out, take-off
    proc_width = [2, 0, 10, 0, 2]         # landing, buffer-in, gate, buffer-out, take-off
    st_list_v = [[1.5, 1, 1, 1],
                [2, 1.5, 1, 1],
                [2.5, 2, 1.5, 1],
                [3, 2.5, 1.5, 1.5]]
    st_list_o = [0.8, 0.6, 0.5, 1.0]
    st_list_r = [1.0, 0.7]         # If you change 'num_pad', you need to also change (len(st_list_r) = num_pad)
    st_list = [
        [  # inner_scale (o)
            [  # v (row)
                [x * inner_scale * outer_scale for outer_scale in st_list_r]  # r
                for x in row
            ]
            for row in st_list_v
        ]
        for inner_scale in st_list_o
    ]
    ready_max = 100.0
    ETA_ready_diff = [2, 10]
    ETD_margin = 5

    return seed, num_ops, num_vehicle, num_pad, num_buffer_in, num_gate, num_buffer_out, num_resource, weights, \
        proc_nominal, proc_width, st_list, ready_max, ETA_ready_diff, ETD_margin


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


def generate_instance(config: InstanceConfig = InstanceConfig()) -> Instance:
    return Instance.from_config(config)
