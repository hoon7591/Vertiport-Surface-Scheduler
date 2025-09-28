import numpy as np
from Scenario import ScenarioConfig, Scenario
from Instance import Instance
from Solver import solve
from Solution import Solution
from VertiportSimulator import VertiportSimulator, VertiportSimulatorRecedingHorizon


def initialize_results_data(scenario_exp):
    start_times_RHC = np.zeros(scenario_exp.num_vehicles, scenario_exp.num_operations)
    finish_times_RHC = np.zeros(scenario_exp.num_vehicles, scenario_exp.num_operations)
    assigned_resources_RHC = np.zeros((scenario_exp.num_vehicles, scenario_exp.num_operations), dtype=int) - 1
    arrival_time_tardiness_RHC = np.zeros(scenario_exp.num_vehicles)
    departure_time_tardiness_RHC = np.zeros(scenario_exp.num_vehicles)

    return start_times_RHC, finish_times_RHC, assigned_resources_RHC, arrival_time_tardiness_RHC, departure_time_tardiness_RHC


def horizon_scheduling(scenario_exp, planning_horizon, processing_vehicles_id):
    instance_in_horizon_exp, activated_vehicle_id_exp, ready_in_horizon_vehicle_id_exp = Scenario.scenario_to_instance(scenario_exp, planning_horizon, processing_vehicles_id)
    solution_in_horizon = solve(instance_in_horizon_exp, solver="exact", is_numerical_exp=False)
    schedule_info_for_run = {}
    for i in range(len(activated_vehicle_id_exp)):
        v_id = activated_vehicle_id_exp[i]
        for j in range(scenario_exp.num_operations):
            schedule_info_for_run[(v_id, j)] = (
                solution_in_horizon.assigned_resources[i, j],
                solution_in_horizon.start_times[i, j]
            )

    return schedule_info_for_run, instance_in_horizon_exp, activated_vehicle_id_exp, ready_in_horizon_vehicle_id_exp


# def horizon_running(scenario_true, schedule_info_for_run, running_horizon, available_time_of_resources, past_pad_info, finish_times_RHC):
#     ready_in_horizon_vehicle_id_true = [[i, 0] for i, x in enumerate(scenario_true.vehicle_arrival_times) if
#                                         running_horizon[0] <= x <= running_horizon[1]]
#     ready_val_in_horizon_true = [scenario_true.vehicle_arrival_times[i] for i in ready_in_horizon_vehicle_id_true]
#
#     mask = (running_horizon[0] <= finish_times_RHC) & (finish_times_RHC <= running_horizon[1])
#     finish_in_horizon_vehicle_id_op_pair_true = np.argwhere(mask).tolist()
#     finish_val_in_horizon_true = finish_times_RHC[mask].tolist()
#
#     runnable_vehicles_info = {}
#     for i in range(len(ready_in_horizon_vehicle_id_true)):
#         runnable_vehicles_info[(ready_in_horizon_vehicle_id_true[i][0],
#                                 ready_in_horizon_vehicle_id_true[i][1])] = ready_val_in_horizon_true[i]
#     for i in range(len(finish_in_horizon_vehicle_id_op_pair_true)):
#         finish_in_horizon_vehicle_id_op_pair_true[i][1] += 1
#         runnable_vehicles_info[(finish_in_horizon_vehicle_id_op_pair_true[i][0],
#                                 finish_in_horizon_vehicle_id_op_pair_true[i][1])] = finish_val_in_horizon_true[i]
#     runnable_vehicles_info_sorted = dict(sorted(runnable_vehicles_info.items(), key=lambda item: item[1]))
#
#     while not all(value > running_horizon[1] for value in runnable_vehicles_info_sorted.values()):

