import copy
from Scenario import ScenarioRHCConfig, Scenario
from Solver import solve
from Solution import Solution
from Instance import Instance
from Visualizer import visualize_gantt, visualize_gantt_plotly, visualize_top5_vehicle_wise_delay
from typing import Any, List
from dataclasses import dataclass, field
import numpy as np
import pickle


def RHC(scenario_RHC_config, scenario_exp, scenario_true, obj_option, is_file_gen) -> Solution:
    current_time = 0.0
    scheduling_horizon = [current_time, current_time + scenario_RHC_config.scheduling_horizon_length]
    update_interval = scenario_RHC_config.update_interval
    scheduler_solving_time_limit = scenario_RHC_config.scheduler_solving_time_limit

    all_assigned_resources_schedule = np.zeros((scenario_exp.num_vehicles, scenario_exp.num_operations), dtype=int) - 1
    all_start_times_schedule = np.zeros((scenario_exp.num_vehicles, scenario_exp.num_operations)) - 1.0
    all_assigned_resources_run = np.zeros((scenario_exp.num_vehicles, scenario_exp.num_operations), dtype=int) - 1
    all_start_times_run = np.zeros((scenario_exp.num_vehicles, scenario_exp.num_operations)) - 1.0

    processing_vehicles_id = []
    processing_vehicles_op = []
    processing_vehicles_res = []
    remaining_proc_time = []

    instance_from_scenario_exp = None
    activated_vehicle_id_exp = []

    while True:
        print(f'\nCurrent Time: {current_time}\n')

        instance_from_scenario_exp, activated_vehicle_id_exp, ready_in_horizon_vehicle_id_exp = Scenario.scenario_to_instance_exp(
            scenario_exp, scenario_true, instance_from_scenario_exp, activated_vehicle_id_exp, scheduling_horizon,
            update_interval, processing_vehicles_id, processing_vehicles_op, remaining_proc_time
        )
        if is_file_gen:
            with open(f'instance_exp_{current_time}.pkl', 'wb') as file:
                pickle.dump(instance_from_scenario_exp, file)
        solution_schedule = solve(instance_from_scenario_exp, solver="exact", is_numerical_exp=True,
                                  processing_vehicles_op=processing_vehicles_op, processing_vehicles_res=processing_vehicles_res,
                                  horizon_start=current_time, obj_option=obj_option, solving_time_limit=scheduler_solving_time_limit)

        if is_file_gen:
            visualize_gantt(solution_schedule, activated_vehicle_id_exp, processing_vehicles_id, processing_vehicles_op, current_time, 'save')
            visualize_gantt_plotly(solution_schedule, activated_vehicle_id_exp, processing_vehicles_id, processing_vehicles_op, current_time, 'save')

        for i in range(len(activated_vehicle_id_exp)):
            for j in range(scenario_exp.num_operations):
                all_start_times_schedule[activated_vehicle_id_exp[i], j] = solution_schedule.start_times[i, j]
                all_assigned_resources_schedule[activated_vehicle_id_exp[i], j] = solution_schedule.assigned_resources[i, j]

        all_start_times_run = copy.deepcopy(all_start_times_schedule)
        all_assigned_resources_run = copy.deepcopy(all_assigned_resources_schedule)
        if 'activated_vehicle_id_true' in locals() or 'activated_vehicle_id_true' in globals():
            for i in range(len(activated_vehicle_id_true)):
                for j in range(scenario_true.num_operations):
                    if round(solution_run.start_times[i, j], 9) <= round(current_time, 9):
                        all_start_times_run[activated_vehicle_id_true[i], j] = solution_run.start_times[i, j]
                        all_assigned_resources_run[activated_vehicle_id_true[i], j] = solution_run.assigned_resources[i, j]

        instance_from_scenario_true, activated_vehicle_id_true = Scenario.scenario_to_instance_true(scenario_true, current_time + update_interval)
        if len(activated_vehicle_id_true) == 0:
            current_time += update_interval
            scheduling_horizon = [current_time, current_time + scenario_RHC_config.scheduling_horizon_length]
            continue
        if is_file_gen:
            with open(f'instance_true_{current_time}.pkl', 'wb') as file:
                pickle.dump(instance_from_scenario_true, file)
            RHC_info = {
                "all_assigned_resources_schedule": all_assigned_resources_schedule,
                "all_start_times_schedule": all_start_times_schedule,
                "all_assigned_resources_run": all_assigned_resources_run,
                "all_start_times_run": all_start_times_run,
                "processing_vehicles_id": processing_vehicles_id,
                "processing_vehicles_op": processing_vehicles_op,
                "processing_vehicles_res": processing_vehicles_res,
                "remaining_proc_time": remaining_proc_time,
                "activated_vehicle_id_exp": activated_vehicle_id_exp,
                "activated_vehicle_id_true": activated_vehicle_id_true
            }
            with open(f'RHC_info_{current_time}.pkl', 'wb') as file:
                pickle.dump(RHC_info, file)
        solution_run = solve(instance_from_scenario_true, solver="run_RHC", is_numerical_exp=True,
                             planned_resource_assignment=all_assigned_resources_run,
                             planned_operation_start_times=all_start_times_run,
                             vehicle_original_id=activated_vehicle_id_true,
                             obj_option=obj_option)

        remaining_proc_time = []

        snapshot = solution_run.snapshot_at_time(current_time + update_interval)
        processing_vehicles_id = snapshot['processing_vehicles_ids']
        processing_vehicles_op = snapshot['processing_operation_ids']
        processing_vehicles_res = snapshot['processing_resource_ids']
        waiting_next_op_vehicles_id = snapshot['waiting_next_op_vehicles_ids']
        waiting_landing_vehicles_id = snapshot['waiting_landing_vehicles_ids']

        for i in range(len(processing_vehicles_id)):
            if processing_vehicles_id[i] not in waiting_next_op_vehicles_id:
                remaining_proc_time.append(scenario_exp.proc[processing_vehicles_op[i]][activated_vehicle_id_true[processing_vehicles_id[i]]]
                                           [processing_vehicles_res[i] - solution_run.resource_ind[processing_vehicles_op[i]][0]]
                                           - (current_time + update_interval - solution_run.start_times[processing_vehicles_id[i], processing_vehicles_op[i]]))
            else:
                remaining_proc_time.append(0.0)
            processing_vehicles_id[i] = activated_vehicle_id_true[processing_vehicles_id[i]]

        if is_file_gen:
            visualize_gantt(solution_run, activated_vehicle_id_true, [], [], current_time, 'save')
            visualize_gantt_plotly(solution_run, activated_vehicle_id_true, [], [], current_time, 'save')

        for i in waiting_landing_vehicles_id:
            scenario_exp.vehicle_arrival_times[activated_vehicle_id_true[i]] = current_time + update_interval

        current_time += update_interval
        scheduling_horizon = [current_time, current_time + scenario_RHC_config.scheduling_horizon_length]

        if len(snapshot['takeoff_finished_vehicles_by_t']) == scenario_exp.num_vehicles:
            visualize_gantt(solution_run, activated_vehicle_id_true, [], [], current_time, 'save')
            visualize_gantt_plotly(solution_run, activated_vehicle_id_true, [], [], current_time, 'save')
            final_solution = solution_run
            break

    return final_solution
