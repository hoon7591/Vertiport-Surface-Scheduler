import copy
import time

from Scenario import ScenarioRHCConfig, Scenario
from Solver import solve
from Solution import Solution
from Instance import Instance
from Visualizer import visualize_gantt, visualize_gantt_plotly, visualize_top5_vehicle_wise_delay
from typing import Any, List
from dataclasses import dataclass, field
import numpy as np
import pickle
from pathlib import Path


def RHC(scenario_RHC_config, scenario_exp, scenario_true, obj_option, is_file_gen, is_result_file_gen, scheduler_runtime_limit, result_dir) -> Solution:
    """
    dir[0]: directory path for saving results
    """

    current_time = 0.0
    scheduling_horizon = [current_time, current_time + scenario_RHC_config.scheduling_horizon_length]
    update_interval = scenario_RHC_config.update_interval

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

    log_data_dir = Path("log_data")
    log_data_dir.mkdir(parents=True, exist_ok=True)

    final_solution = None
    num_is_runtime_over_true = 0
    num_is_solution_exist_false = 0
    num_is_infeasible_true = 0

    while True:
        print(f'\nCurrent Time: {current_time}\n')

        instance_from_scenario_exp, activated_vehicle_id_exp, ready_in_horizon_vehicle_id_exp = Scenario.scenario_to_instance_exp(
            scenario_exp, scenario_true, instance_from_scenario_exp, activated_vehicle_id_exp, scheduling_horizon,
            update_interval, processing_vehicles_id, processing_vehicles_op, remaining_proc_time
        )
        
        if is_file_gen:
            path = log_data_dir / f"instance_exp_{current_time}.pkl"
            with path.open('wb') as file:
                pickle.dump(instance_from_scenario_exp, file)

        solution_schedule = solve(instance_from_scenario_exp, solver="exact", is_numerical_exp=True,
                                  processing_vehicles_op=processing_vehicles_op, processing_vehicles_res=processing_vehicles_res,
                                  horizon_start=current_time, obj_option=obj_option, scheduler_runtime_limit=scheduler_runtime_limit)

        num_is_runtime_over_true += int(solution_schedule.is_runtime_over is True)
        num_is_solution_exist_false += int(solution_schedule.is_solution_exist is False)
        num_is_infeasible_true += int(solution_schedule.is_infeasible is True)
        if solution_schedule.is_solution_exist is False and solution_schedule.is_infeasible is False:
            print(f"Scheduling at time {current_time} exceeded runtime limit of {scheduler_runtime_limit} seconds. Reschedule that instance.")
            time.sleep(60.0)
            continue
        elif solution_schedule.is_infeasible is True:
            print(f"Scheduling at time {current_time} is infeasible. Break")
            break

        if is_file_gen:
            path = log_data_dir / f"solution_schedule_{current_time}.pkl"
            with path.open('wb') as file:
                pickle.dump(solution_schedule, file)
            visualize_gantt(solution_schedule, activated_vehicle_id_exp, processing_vehicles_id, processing_vehicles_op, current_time, 'save', log_data_dir)
            visualize_gantt_plotly(solution_schedule, activated_vehicle_id_exp, processing_vehicles_id, processing_vehicles_op, current_time, 'save', current_time, log_data_dir)

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
            path = log_data_dir / f"instance_true_{current_time}.pkl"
            with path.open('wb') as file:
                pickle.dump(instance_from_scenario_true, file)
            path = log_data_dir / f"RHC_info_{current_time}.pkl"
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
            with path.open('wb') as file:
                pickle.dump(RHC_info, file)

        solution_run = solve(instance_from_scenario_true, solver="run_RHC", is_numerical_exp=True,
                             planned_resource_assignment=all_assigned_resources_run,
                             planned_operation_start_times=all_start_times_run,
                             vehicle_original_id=activated_vehicle_id_true,
                             obj_option=obj_option,
                             next_current_time=current_time + update_interval
                             )

        remaining_proc_time = []

        snapshot = solution_run.snapshot_at_time(current_time + update_interval)
        processing_vehicles_id = snapshot['processing_vehicles_ids']
        processing_vehicles_op = snapshot['processing_operation_ids']
        processing_vehicles_res = snapshot['processing_resource_ids']
        waiting_next_op_vehicles_id = snapshot['waiting_next_op_vehicles_ids']
        waiting_landing_vehicles_id = snapshot['waiting_landing_vehicles_ids']

        for i in range(len(processing_vehicles_id)):
            if processing_vehicles_id[i] not in waiting_next_op_vehicles_id:
                remaining_proc_time.append(max(scenario_exp.proc[processing_vehicles_op[i]][activated_vehicle_id_true[processing_vehicles_id[i]]][processing_vehicles_res[i] - solution_run.resource_ind[processing_vehicles_op[i]][0]]
                                           - (current_time + update_interval - solution_run.start_times[processing_vehicles_id[i], processing_vehicles_op[i]]), 0.0))
                if activated_vehicle_id_true[processing_vehicles_id[i]] in scenario_RHC_config.dynamic_proc_v_id and remaining_proc_time[-1] <= update_interval:
                    idx = scenario_RHC_config.dynamic_proc_v_id.index(activated_vehicle_id_true[processing_vehicles_id[i]])
                    if processing_vehicles_op[i] == scenario_RHC_config.dynamic_proc_op[idx]:
                        remaining_proc_time[-1] += scenario_RHC_config.dynamic_proc_inc_time[idx]
                # if remaining_proc_time[-1] == 0.0:
                #     remaining_proc_time[-1] = max(scenario_true.proc[processing_vehicles_op[i]][activated_vehicle_id_true[processing_vehicles_id[i]]][processing_vehicles_res[i] - solution_run.resource_ind[processing_vehicles_op[i]][0]]
                #                                   - (current_time + update_interval - solution_run.start_times[processing_vehicles_id[i], processing_vehicles_op[i]]), 0.0)
            else:
                remaining_proc_time.append(0.0)
            processing_vehicles_id[i] = activated_vehicle_id_true[processing_vehicles_id[i]]

        if is_file_gen:
            path = log_data_dir / f"solution_run_{current_time}.pkl"
            with path.open('wb') as file:
                pickle.dump(solution_run, file)
            visualize_gantt(solution_run, activated_vehicle_id_true, [], [], current_time, 'save', log_data_dir)
            visualize_gantt_plotly(solution_run, activated_vehicle_id_true, [], [], current_time, 'save', current_time, log_data_dir)

        for i in waiting_landing_vehicles_id:
            scenario_exp.vehicle_arrival_times[activated_vehicle_id_true[i]] = current_time + update_interval

        current_time += update_interval
        scheduling_horizon = [current_time, current_time + scenario_RHC_config.scheduling_horizon_length]

        if len(snapshot['takeoff_finished_vehicles_by_t']) == scenario_exp.num_vehicles:
            final_solution = solution_run

            v_id_nominal_proc = list(range(scenario_exp.num_vehicles))
            dynamic_proc_v_id_set = set(scenario_RHC_config.dynamic_proc_v_id)
            v_id_nominal_proc = [v_id for v_id in v_id_nominal_proc if v_id not in dynamic_proc_v_id_set]
            vehicle_wise_weighted_tardiness_nominal = final_solution.vehicle_wise_weighted_tardiness[v_id_nominal_proc]
            final_solution.total_weighted_sum_tardiness_nominal = np.sum(vehicle_wise_weighted_tardiness_nominal)
            final_solution.max_arrival_tardiness_nominal = np.max(final_solution.arrival_time_tardiness[v_id_nominal_proc])
            final_solution.max_departure_tardiness_nominal = np.max(final_solution.departure_time_tardiness[v_id_nominal_proc])
            final_solution.max_vehicle_wise_weighted_tardiness_nominal = np.max(vehicle_wise_weighted_tardiness_nominal)
            final_solution.std_vehicle_wise_weighted_tardiness_nominal = np.std(vehicle_wise_weighted_tardiness_nominal)
            final_solution.cv_vehicle_wise_weighted_tardiness_nominal = final_solution.std_vehicle_wise_weighted_tardiness_nominal / np.mean(
                vehicle_wise_weighted_tardiness_nominal) if np.mean(vehicle_wise_weighted_tardiness_nominal) > 0 else 0
            final_solution.gini_vehicle_wise_weighted_tardiness_nominal = (np.sum(
                np.abs(vehicle_wise_weighted_tardiness_nominal[:, None] - vehicle_wise_weighted_tardiness_nominal)) / (
                                                                                 2 * scenario_exp.num_vehicles * np.sum(
                                                                             vehicle_wise_weighted_tardiness_nominal))) if scenario_exp.num_vehicles > 0 and np.mean(
                vehicle_wise_weighted_tardiness_nominal) > 0 else 0

            if is_result_file_gen:
                if result_dir is not None:
                    path = result_dir / f"RHC_{obj_option}_final_solution.pkl"
                    with path.open('wb') as file:
                        pickle.dump(final_solution, file)
                    visualize_gantt(final_solution, activated_vehicle_id_true, [], [], current_time, 'save', result_dir)
                    visualize_gantt_plotly(final_solution, activated_vehicle_id_true, [], [], current_time, 'save', current_time, result_dir)
                else:
                    with open(f'RHC_{obj_option}_final_solution.pkl', 'wb') as file:
                        pickle.dump(final_solution, file)
                    visualize_gantt(final_solution, activated_vehicle_id_true, [], [], current_time, 'save')
                    visualize_gantt_plotly(final_solution, activated_vehicle_id_true, [], [], current_time, 'save')
            break

    return final_solution, num_is_runtime_over_true, num_is_solution_exist_false, num_is_infeasible_true
