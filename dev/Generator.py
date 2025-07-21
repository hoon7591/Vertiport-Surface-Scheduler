import numpy as np


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


def generate_instance(seed, num_ops, num_vehicle, num_pad, num_buffer_in, num_gate, num_buffer_out, proc_nominal,
                      proc_width, st_list, ready_max, ETA_ready_diff, ETD_margin):
    np.random.seed(seed)

    ready = np.random.uniform(low=0.0, high=ready_max, size=num_vehicle)

    proc_landing = np.random.uniform(low=proc_nominal[0] - proc_width[0] / 2, high=proc_nominal[0] + proc_width[0] / 2,
                                     size=(num_vehicle, num_pad))
    proc_buffer_in = np.zeros((num_vehicle, num_buffer_in))
    # proc_buffer_in = np.random.uniform(low=proc_nominal[1] - proc_width[1] / 2, high=proc_nominal[1] + proc_width[1] / 2,
    #                                    size=(num_vehicle, num_buffer_in))
    proc_gate = np.zeros((num_vehicle, num_gate))
    TAT = np.zeros(num_vehicle)
    for i in range(num_vehicle):
        proc_gate[i][0] = np.random.uniform(low=proc_nominal[2] - proc_width[2] / 2, high=proc_nominal[2] + proc_width[2] / 2)
        TAT[i] = proc_gate[i][0]
    for i in range(num_vehicle):
        for j in range(num_gate):
            proc_gate[i][j] = proc_gate[i][0] + 0.15 * (j + 1)
    proc_buffer_out = np.zeros((num_vehicle, num_buffer_out))
    # proc_buffer_out = np.random.uniform(low=proc_nominal[3]-proc_width[3]/2, high=proc_nominal[3]+proc_width[3]/2,
    #                                     size=(num_vehicle, num_buffer_out))
    proc_takeoff = np.random.uniform(low=proc_nominal[4] - proc_width[4] / 2, high=proc_nominal[4] + proc_width[4] / 2,
                                     size=(num_vehicle, num_pad))
    proc = [proc_landing, proc_buffer_in, proc_gate, proc_buffer_out, proc_takeoff]

    ETA_ready_diff_arr = ETA_ready_diff[0] + ETA_ready_diff[0] / 2 * np.random.randn(num_vehicle)
    due_a = ready + ETA_ready_diff_arr         # ETA

    due_d = due_a + TAT + ETD_margin            # ETD

    vehicle_type = np.random.randint(0, 4, num_vehicle)

    # Map o indices (0,1,2,3) to new keys
    o_key_map = {
        0: (0, 0),
        1: (0, num_ops - 1),
        2: (num_ops - 1, 0),
        3: (num_ops - 1, num_ops - 1)
    }
    ST = {}  # New dict with keys as tuples (1,1), (1,5), ...
    for o in range(len(st_list)):
        key = o_key_map[o]
        ST[key] = []
        for i in range(num_vehicle):
            row = []
            v_i = vehicle_type[i]
            for j in range(num_vehicle):
                v_j = vehicle_type[j]
                row.append(st_list[o][v_i][v_j])  # This is a list of length len(st_list_r)
            ST[key].append(row)

    def round_nested_list(lst, digits=2):
        if isinstance(lst, list):
            return [round_nested_list(x, digits) for x in lst]
        elif isinstance(lst, float):
            return round(lst, digits)
        else:
            return lst

    ST_rounded = {k: round_nested_list(v, digits=2) for k, v in ST.items()}

    # M = proc_landing.sum() + proc_gate.sum() + proc_takeoff.sum()
    M = ready_max + proc_landing.sum() / num_pad + proc_gate.sum() / num_gate + proc_takeoff.sum() / num_pad

    return ready, proc, due_a, due_d, ST_rounded, vehicle_type, M

# seed, num_ops, num_vehicle, num_pad, num_buffer_in, num_gate, num_buffer_out, num_resource, weights, proc_nominal, proc_width, st_list, ready_max, ETA_ready_diff, ETD_margin = hyper_param_setting()
# ready, proc, due_a, due_d, ST, vehicle_type, M = generate_instance(seed, num_ops, num_vehicle, num_pad, num_buffer_in, num_gate, num_buffer_out, proc_nominal, proc_width, st_list, ready_max, ETA_ready_diff, ETD_margin)
# print(ST)
