from Generator import generate_instance
from Solver import solve
from Visualizer import visualize_result
from Instance import InstanceConfig


"""
Hyper Parameter Lists for Problem Generation in InstanceConfig
seed: for fixing random seed
num_ops: number of operations in the vertiport service (depending on the existence of buffer_in and buffer_out, the num_ops can be 3, 4, or 5)
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
solver: option for solver selction
        - "exact": exact solver which guarantees optimal solution
        - "FCFS_Gurobi": First-Come-First-Served sequencing is applied to Gurobi solver
        - "FCFS_landing_Gurobi": Gurobi solver which applies First-Come-First-Served logic only for landing operations
"""


"""
1. Time unit is minute
2. If you change 'num_pad', you need to also change 'st_list_r' => len(st_list_r) = num_pad
"""


if __name__ == "__main__":
    print("This is a module for solving optimization problems using Gurobi.")
    config = InstanceConfig(unified_buffer=True)  # Example: override defaults
    instance = generate_instance(config)
    solution = solve(instance, solver="exact")
    visualize_result(instance, solution)
