from gurobipy import GRB, Model, quicksum
import numpy as np
from Solution import Solution
from Instance import Instance


def solve(instance: Instance, solver) -> Solution:
    seed = instance.seed
    num_ops = instance.num_ops
    num_vehicle = instance.num_vehicle
    num_pad = instance.num_pad
    num_buffer_in = instance.num_buffer_in
    num_gate = instance.num_gate
    num_buffer_out = instance.num_buffer_out
    num_resource = instance.num_resource
    weights = instance.weights
    ready = instance.ready
    proc = instance.proc
    due_a = instance.due_a
    due_d = instance.due_d
    ST = instance.ST
    M = instance.M
    unified_buffer = instance.unified_buffer

    # Initialize Model
    model = Model("Vertiport_Surface_Scheduler")
    model.setParam('Seed', seed)
    model.setParam("OutputFlag", 1)
    # model.setParam("MIPGap", 0.01)  # 1% gap tolerance (for Gurobi mode)
    # model.setParam("TimeLimit", 60)  # 60 seconds (1 minute) (for Gurobi mode)
    # if solver == "FCFS" or solver == "FCFS_landing" or solver == "no_rule":
    #     model.setParam("PoolSearchMode", 2)  # Do a systematic search for alternatives (for SAT mode)
    #     model.setParam("PoolSolutions", 10)  # Search for up to 10 feasible solutions (for SAT mode)

    # Define Variables
    S = model.addVars(num_ops, num_vehicle, vtype=GRB.CONTINUOUS, lb=0, name="start_time")           # S[o, v]
    y = model.addVars(num_ops, num_vehicle, num_resource, vtype=GRB.BINARY, name="y")         # y[o, v, r]
    x = model.addVars(num_ops, num_ops, num_vehicle, num_vehicle, num_resource, vtype=GRB.BINARY, name="x")          # x[o, o', v, v', r]
    T_a = model.addVars(num_vehicle, vtype=GRB.CONTINUOUS, lb=0, name="tardiness_arrival")
    T_d = model.addVars(num_vehicle, vtype=GRB.CONTINUOUS, lb=0, name="tardiness_departure")

    # Define Objective
    if solver == "FCFS" or solver == "FCFS_landing" or solver == "no_rule":
        pass
    else:
        model.setObjective(quicksum(weights[0] * T_a[i] + weights[1] * T_d[i] for i in range(num_vehicle)), GRB.MINIMIZE)

    # Set Constraints
    if unified_buffer:
        if num_buffer_in == 0:
            resource_ind = [[0, num_pad], [num_pad, num_pad + num_gate], [0, num_pad]]  # [landing, gate, takeoff]
        else:
            resource_ind = [[0, num_pad], [num_pad, num_pad + num_buffer_in],
                           [num_pad + num_buffer_in, num_pad + num_buffer_in + num_gate],
                           [num_pad, num_pad + num_buffer_in],
                           [0, num_pad]]  # [landing, buffer_in, gate, buffer_out, takeoff]
    else:
        if num_buffer_in == 0:
            if num_buffer_out == 0:
                resource_ind = [[0, num_pad], [num_pad, num_pad + num_gate], [0, num_pad]]  # [landing, gate, takeoff]
            else:
                resource_ind = [[0, num_pad], [num_pad, num_pad + num_gate],
                                [num_pad + num_gate, num_pad + num_gate + num_buffer_out], [0, num_pad]]    # [landing, gate, buffer_out, takeoff]
        else:
            if num_buffer_out == 0:
                resource_ind = [[0, num_pad], [num_pad, num_pad + num_buffer_in],
                                [num_pad + num_buffer_in, num_pad + num_buffer_in + num_gate], [0, num_pad]]    # [landing, buffer_in, gate, takeoff]
            else:
                resource_ind = [[0, num_pad], [num_pad, num_pad + num_buffer_in],
                                [num_pad + num_buffer_in, num_pad + num_buffer_in + num_gate],
                                [num_pad + num_buffer_in + num_gate, num_pad + num_buffer_in + num_gate + num_buffer_out],
                                [0, num_pad]]  # [landing, buffer_in, gate, buffer_out, takeoff]

    # Const. 1: Vehicle Assignment
    for i in range(num_ops):
        for j in range(num_vehicle):
            model.addConstr(quicksum(y[i, j, k] for k in range(resource_ind[i][0], resource_ind[i][1])) == 1)

    # Const. 2: Precedence
    for i in range(num_ops - 1):
        for j in range(num_vehicle):
            model.addConstr(S[i + 1, j] >= S[i, j] + quicksum(y[i, j, k] * proc[i][j][k - resource_ind[i][0]]
                                                              for k in range(resource_ind[i][0], resource_ind[i][1])))

    # Const. 3: Overlap Avoidance with Reentrant Condition and Blocking
    for i in range(num_ops):
        for i_ in range(num_ops):
            res_i = list(range(resource_ind[i][0], resource_ind[i][1]))
            res_i_ = list(range(resource_ind[i_][0], resource_ind[i_][1]))
            res_intersect = list(set(res_i) & set(res_i_))
            if len(res_intersect) == 0:
                continue
            for j in range(num_vehicle):
                for j_ in range(num_vehicle):
                    if j != j_:
                        for k in res_intersect:
                            model.addConstr(x[i, i_, j, j_, k] <= y[i, j, k])
                            model.addConstr(x[i, i_, j, j_, k] <= y[i_, j_, k])
                            model.addConstr(x[i, i_, j, j_, k] + x[i_, i, j_, j, k] <= 1)
                            model.addConstr(x[i, i_, j, j_, k] + x[i_, i, j_, j, k] >= y[i, j, k] + y[i_, j_, k] - 1)
                            if i != num_ops - 1:
                                model.addConstr(S[i + 1, j] <= S[i_, j_] + M * (1 - x[i, i_, j, j_, k]))
                            else:
                                model.addConstr(S[i, j] + proc[i][j][k - resource_ind[i][0]] <= S[i_, j_] + M * (1 - x[i, i_, j, j_, k]))

                            # Const. 4: Separation
                            if (i == 0 or i == num_ops - 1) and (i_ == 0 or i_ == num_ops - 1):
                                model.addConstr(S[i_, j_] >= S[i, j] + proc[i][j][k - resource_ind[i][0]] +
                                                ST[(i, i_)][j][j_][k] - M * (1 - x[i, i_, j, j_, k]))

    # Const. 5: Ready
    for i in range(num_vehicle):
        model.addConstr(S[0, i] >= ready[i])

    # Const. 6: Tardiness Calculation with No early Departure
    for i in range(num_vehicle):
        model.addConstr(T_a[i] >= S[0, i] + quicksum(y[0, i, j] * proc[0][i][j - resource_ind[0][0]]
                                                     for j in range(resource_ind[0][0], resource_ind[0][1])) - due_a[i])
        model.addConstr(S[num_ops - 1, i] >= due_d[i])
        model.addConstr(T_d[i] == S[num_ops - 1, i] - due_d[i])

    # Const. 7: FCFS Implementation
    if solver == "FCFS_Gurobi" or solver == "FCFS":
        sorted_vehicle = np.argsort(ready)
        for i in range(num_ops):
            for j in range(num_vehicle - 1):
                model.addConstr(S[i, sorted_vehicle[j]] <= S[i, sorted_vehicle[j + 1]])
        for i in range(num_ops):
            for j in range(num_vehicle - 1):
                for j_ in range(j + 1, num_vehicle):
                    for k in range(resource_ind[i][0], resource_ind[i][1]):
                        model.addConstr(x[i, i, sorted_vehicle[j_], sorted_vehicle[j], k] == 0)
    elif solver == "FCFS_landing_Gurobi" or solver == "FCFS_landing":
        sorted_vehicle = np.argsort(ready)
        for j in range(num_vehicle - 1):
            model.addConstr(S[0, sorted_vehicle[j]] <= S[0, sorted_vehicle[j + 1]])
        for j in range(num_vehicle - 1):
            for j_ in range(j + 1, num_vehicle):
                for k in range(resource_ind[0][0], resource_ind[0][1]):
                    model.addConstr(x[0, 0, sorted_vehicle[j_], sorted_vehicle[j], k] == 0)

    # Solve Model
    model.optimize()

    # Extract Solution
    start_time_arr = np.zeros((num_vehicle, num_ops))
    finish_time_arr = np.zeros((num_vehicle, num_ops))
    assigned_res_arr = np.zeros((num_vehicle, num_ops))
    arrival_tar_arr = np.zeros(num_vehicle)
    departure_tar_arr = np.zeros(num_vehicle)

    Obj = model.ObjVal
    Gurobi_Runtime = model.Runtime

    for i in range(num_vehicle):
        arrival_tar_arr[i] = T_a[i].X
        departure_tar_arr[i] = T_d[i].X
        for j in range(num_ops):
            start_time_arr[i, j] = S[j, i].X
            finish_time_arr[i, j] = start_time_arr[i, j]
            for k in range(resource_ind[j][0], resource_ind[j][1]):
                if y[j, i, k].X >= 0.5:
                    finish_time_arr[i, j] += proc[j][i][k - resource_ind[j][0]]
                    assigned_res_arr[i, j] = k

    if solver == "FCFS" or solver == "FCFS_landing" or solver == "no_rule":
        Obj = weights[0] * sum(arrival_tar_arr) + weights[1] * sum(departure_tar_arr)

    return Solution(Obj, Gurobi_Runtime, start_time_arr, finish_time_arr, assigned_res_arr, 
                   arrival_tar_arr, departure_tar_arr, resource_ind, solver, instance)
