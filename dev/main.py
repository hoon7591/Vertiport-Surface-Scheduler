from Generator import generate_instance
from Solver import solve
from Visualizer import visualize_result
from Instance import InstanceConfig


# Hyper Parameter Lists for Problem Generation in InstanceConfig

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

# solver: option for solver selction
#         - "exact": exact solver which guarantees optimal solution
#         - "FCFS_Gurobi": First-Come-First-Served sequencing is applied to Gurobi solver
#         - "FCFS_landing_Gurobi": Gurobi solver which applies First-Come-First-Served logic only for landing operations
#         - "FCFS": First-Come-First-Served sequencing is applied to SAT solver
#         - "FCFS_landing": SAT solver which applies First-Come-First-Served logic only for landing operations
#         - "no_rule": SAT solver without any primary sequencing rule

# Notes:
# 1. Time unit is minute
# 2. If you change 'num_pad', you need to also change 'st_list_r' => len(st_list_r) = num_pad


if __name__ == "__main__":
    print("This is a module for solving optimization problems using Gurobi.")
    config = InstanceConfig(unified_buffer=True)  # Example: override defaults
    instance = generate_instance(config)
    solution = solve(instance, solver="FCFS")
    visualize_result(solution)
