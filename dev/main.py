from Generator import generate_instance, generate_scenario_exp
from Solver import solve
from Visualizer import visualize_gantt, visualize_top5_vehicle_wise_delay, visualize_gantt_plotly
from Instance import InstanceConfig
from VertiportSimulator import VertiportSimulator
from Numerical_Study import ExperimentConfig, Numerical_Experiment
from Scenario import ScenarioRHCConfig, Scenario
from RecedingHorizonController import RHC
import numpy as np
import pickle

"""
Hyper Parameter Lists for Problem Generation in InstanceConfig

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

solver: option for solver selection
        - "exact": exact solver which guarantees optimal solution
        - "FCFS_Gurobi": First-Come-First-Served sequencing is applied to Gurobi solver
        - "FCFS_landing_Gurobi": Gurobi solver which applies First-Come-First-Served logic only for landing operations
        - "FCFS_SAT": First-Come-First-Served sequencing is applied to SAT solver
        - "FCFS_landing_SAT": SAT solver which applies First-Come-First-Served logic only for landing operations
        - "no_rule_SAT": SAT solver without any primary sequencing rule
        - "FCFS_heuristic": First-Come-First-Served heuristic solver (Discrete Event Simulation)

is_numerical_exp: flag for numerical experiment mode or solving mode for single instance

obj_option: option for objective function selection (activated only for the solvers which have objective function)
        - "weighted_sum": weighted sum of arrival delay and departure delay
        - "vehicle_wise_max": vehicle-wise max value of weighted sum of arrival delay and departure delay
        - "weighted_sum_of_max": weighted sum of max value of arrival delay and departure delay

Notes:
1. Time unit is minute
2. If you change 'num_pad', you need to also change 'st_list_r' => len(st_list_r) = num_pad
3. If you want to construct 'unified_buffer' model, you need to insert 'num_buffer' only,
   otherwise, you need to insert 'num_buffer_in' and 'num_buffer_out'
"""


if __name__ == "__main__":

    #################### Single Instance Generation ####################
    # config = InstanceConfig(is_unified_buffer=True, num_vehicles=18, num_buffer=2, num_gate=6, num_pad=2)
    # # config = InstanceConfig(is_unified_buffer=False, num_vehicles=15, num_buffer_in=2, num_gate=6, num_pad=2, num_buffer_out=2,
    # #                         ETD_margin=3, gate_close_margin=2, ETA_ready_diff=[3, 2], seed=27)  # Example: override defaults
    # # config = InstanceConfig(is_unified_buffer=True, num_vehicles=15, num_buffer=2, num_gate=6, num_pad=2,
    # #                         ETD_margin=3, gate_close_margin=2, ETA_ready_diff=[3, 2], seed=20)
    # # config = InstanceConfig(is_unified_buffer=True, num_vehicles=15, num_buffer=2, num_gate=6, num_pad=2,
    # #                         ETD_margin=3, gate_close_margin=2, ETA_ready_diff=[3, 2], seed=65)
    # # config = InstanceConfig(is_unified_buffer=True, num_vehicles=15, num_buffer=2, num_gate=6, num_pad=2,
    # #                         ETD_margin=4, gate_close_margin=2.5, ETA_ready_diff=[3, 2], seed=8)
    # # config = InstanceConfig(is_unified_buffer=False, num_vehicles=50, num_buffer_in=2, num_buffer_out=2, num_gate=6, num_pad=2,
    # #                         ETD_margin=5, gate_close_margin=3, ETA_ready_diff=[3, 6], seed=174)
    # # config = InstanceConfig(is_unified_buffer=True, num_vehicles=50, num_buffer=6, num_gate=18, num_pad=6)
    # # config = InstanceConfig(is_unified_buffer=True, num_vehicles=9, num_buffer=3, num_gate=9, num_pad=3, ready_max=8)
    # instance = generate_instance(config)

    #################### Solver Execution for Single Instance ####################
    # solution = solve(instance, solver="exact", is_numerical_exp=True, obj_option="weighted_sum")
    # visualize_gantt(solution, list(range(instance.num_vehicles)), [], [], None, 'show')
    # visualize_top5_vehicle_wise_delay(solution, list(range(instance.num_vehicles)))

    #################### Numerical Study ####################
    # exp_config = ExperimentConfig(num_vehicles_pad_buffer_gate_exp=[[10, 14, 2, 6], [15, 21, 3, 9], [20, 28, 4, 12]],
    #                               ETD_margin_exp=[3, 5], ETD_margin_minus_NED_exp=[1, 2],
    #                               ETA_ready_diff_sigma_exp=[1, 3, 5], is_unified_buffer_exp=[True, False])
    # Numerical_Experiment(exp_config)

    #################### Scenario Generation ####################
    ### For test of Run_RHC by setting std as zero (same results between run and solve are expected) ###
    # scenario_RHC_config = ScenarioRHCConfig(scheduling_horizon_length=30.0, update_interval=1.0, operation_hour=18,
    #                                         num_vehicles_per_hour=10, disturbance_std_proc=[0.0, 0.0, 0.0],
    #                                         disturbance_std_ready=0.0, is_unified_buffer=True, num_pad=2,
    #                                         num_buffer=2, num_gate=8)
    ### For test with non-zero std (different results between run and solve are expected) ###
    scenario_RHC_config = ScenarioRHCConfig(scheduling_horizon_length=30.0, update_interval=1.0, operation_hour=18,
                                            num_vehicles_per_hour=10, disturbance_std_proc=[0.3, 0.5, 0.3],
                                            disturbance_std_ready=1.0, is_unified_buffer=True, num_pad=2,
                                            num_buffer=2, num_gate=8)
    ### For test with dynamic arrival and processing ###
    # scenario_RHC_config = ScenarioRHCConfig(scheduling_horizon_length=30.0, update_interval=1.0, operation_hour=18,
    #                                         num_vehicles_per_hour=10, disturbance_std_proc=[0.3, 0.5, 0.3],
    #                                         disturbance_std_ready=5.0, is_unified_buffer=True, num_pad=2,
    #                                         num_buffer=2, num_gate=8,
    #                                         dynamic_arrival_v_id=[98, 77, 66, 111, 71, 61, 138, 225, 89, 48, 93, 38, 2, 7, 52],
    #                                         dynamic_proc_v_id=[99, 117, 78, 8, 35, 150],
    #                                         dynamic_proc_op=[2, 2, 2, 4, 0, 2],
    #                                         dynamic_proc_inc_time=[10.0, 15.0, 10.0, 5.0, 7.0, 20.0])
    scenario_exp = Scenario.from_scenario_config_exp(scenario_RHC_config)
    scenario_true = Scenario.from_scenario_config_true(scenario_RHC_config, scenario_exp)

    #################### RHC Test ####################
    # instance_from_scenario_exp = None
    # activated_vehicle_id_exp = []
    #
    # instance_from_scenario_exp, activated_vehicle_id_exp, ready_in_horizon_vehicle_id_exp = Scenario.scenario_to_instance_exp(scenario_exp, scenario_true, instance_from_scenario_exp, activated_vehicle_id_exp, [0.0, 100.0], 100.0, [], [], [])
    # solution = solve(instance_from_scenario_exp, solver="exact", is_numerical_exp=False)
    # # visualize_gantt(solution, activated_vehicle_id_exp, [], [], None, 'show')
    # visualize_gantt_plotly(solution, activated_vehicle_id_exp, [], [], None, 'show')
    #
    # instance_from_scenario_exp, activated_vehicle_id_exp, ready_in_horizon_vehicle_id_exp = Scenario.scenario_to_instance_exp(scenario_exp, [20.0, 40.0], 3.0, [171, 208, 244, 205, 72, 128], [0, 2, 2, 2, 2, 2], [20.3, 22.5, 36.6, 33.2, 36.7, 26.8])
    # solution = solve(instance_from_scenario_exp, solver="exact_RHC", is_numerical_exp=False, processing_vehicles_op=[0, 2, 2, 2, 2, 2], processing_vehicles_res=[1, 4, 5, 6, 8, 9], horizon_start=20.0)
    # visualize_gantt(solution, activated_vehicle_id_exp, [171, 208, 244, 205, 72, 128], [0, 2, 2, 2, 2, 2], None, 'show')
    #
    # instance_from_scenario_exp, activated_vehicle_id_exp, ready_in_horizon_vehicle_id_exp = Scenario.scenario_to_instance_exp(scenario_exp, [40.0, 60.0], 3.0, [42, 145, 237, 98, 10, 100, 72, 171], [0, 0, 2, 2, 2, 2, 2, 2], [40.0, 42.2, 55.5, 51.4, 42.3, 54.8, 40.0, 40.9])
    # solution = solve(instance_from_scenario_exp, solver="exact_RHC", is_numerical_exp=False, processing_vehicles_op=[0, 0, 2, 2, 2, 2, 2, 2], processing_vehicles_res=[0, 1, 4, 5, 9, 10, 11, 13], horizon_start=40.0)
    # visualize_gantt(solution, activated_vehicle_id_exp, [42, 145, 237, 98, 10, 100, 72, 171], [0, 0, 2, 2, 2, 2, 2, 2], None, 'show')
    #
    # instance_from_scenario_exp, activated_vehicle_id_exp, ready_in_horizon_vehicle_id_exp = Scenario.scenario_to_instance_exp(scenario_exp, [60.0, 80.0], 3.0, [237, 29, 42, 145, 58, 168, 148], [4, 2, 2, 2, 2, 2, 2], [62.1, 80.2, 66.4, 60.5, 79.8, 75.9, 83.9])
    # solution = solve(instance_from_scenario_exp, solver="exact_RHC", is_numerical_exp=False, processing_vehicles_op=[4, 2, 2, 2, 2, 2, 2], processing_vehicles_res=[1, 5, 6, 7, 8, 9, 12], horizon_start=60.0)
    # visualize_gantt(solution, activated_vehicle_id_exp, [237, 29, 42, 145, 58, 168, 148], [4, 2, 2, 2, 2, 2, 2], None, 'show')
    #
    # instance_from_scenario_exp, activated_vehicle_id_exp, ready_in_horizon_vehicle_id_exp = Scenario.scenario_to_instance_exp(scenario_exp, [80.0, 100.0], 3.0, [77, 29, 32, 58, 6, 148, 83], [0, 2, 2, 2, 2, 2, 2], [82.8, 80.2, 91.6, 80.0, 84.0, 83.9, 90.0])
    # solution = solve(instance_from_scenario_exp, solver="exact_RHC", is_numerical_exp=False, processing_vehicles_op=[0, 2, 2, 2, 2, 2, 2], processing_vehicles_res=[0, 5, 7, 8, 11, 12, 13], horizon_start=80.0)
    # visualize_gantt(solution, activated_vehicle_id_exp, [77, 29, 32, 58, 6, 148, 83], [0, 2, 2, 2, 2, 2, 2], None, 'show')

    instance_from_scenario_true, activated_vehicle_id_true = Scenario.scenario_to_instance_true(scenario_true, 1200.0)
    solution = solve(instance_from_scenario_true, solver="FCFS_heuristic", is_numerical_exp=True)
    visualize_gantt(solution, activated_vehicle_id_true, [], [], None, 'save')
    visualize_gantt_plotly(solution, activated_vehicle_id_true, [], [], None, 'save')

    # instance_from_scenario_exp, activated_vehicle_id_exp, ready_in_horizon_vehicle_id_exp = Scenario.scenario_to_instance_exp(scenario_exp, [0.0, 50.0], 3.0, [], [], [])
    # solution = solve(instance_from_scenario_exp, solver="exact", is_numerical_exp=False)
    # visualize_gantt(solution, activated_vehicle_id_exp, [], [], None, 'show')
    #
    # instance_from_scenario_true, activated_vehicle_id_true = Scenario.scenario_to_instance_true(scenario_true, 100.0)
    # solution = solve(instance_from_scenario_true, solver="run_RHC", is_numerical_exp=True,
    #                  planned_resource_assignment=solution.assigned_resources,
    #                  planned_operation_start_times=solution.start_times,
    #                  vehicle_original_id=list(range(len(activated_vehicle_id_exp))))
    # # visualize_gantt(solution, activated_vehicle_id_true, [], [], None, 'show')
    # visualize_gantt_plotly(solution, activated_vehicle_id_true, [], [], None, 'show')
    #
    # instance_from_scenario_exp, activated_vehicle_id_exp, ready_in_horizon_vehicle_id_exp = Scenario.scenario_to_instance_exp(scenario_exp, scenario_true, [40.0, 90.0], 3.0, [42, 145, 72, 98, 171, 237, 10, 100], [0, 0, 3, 2, 2, 2, 2, 2], [40.03, 42.2, 40.0, 53.8, 41.8, 55.4, 43.8, 55.5])
    # solution = solve(instance_from_scenario_exp, solver="exact_RHC", is_numerical_exp=False, processing_vehicles_op=[0, 0, 3, 2, 2, 2, 2, 2], processing_vehicles_res=[0, 1, 3, 5, 6, 7, 8, 9], horizon_start=40.0, obj_option="weighted_sum")
    # visualize_gantt(solution, activated_vehicle_id_exp, [42, 145, 72, 98, 171, 237, 10, 100], [0, 0, 3, 2, 2, 2, 2, 2], None, 'show')

    # with open('instance_true_985.0.pkl', 'rb') as file:
    #     instance_from_scenario_true = pickle.load(file)
    # with open('RHC_info_985.0.pkl', 'rb') as file:
    #     RHC_info = pickle.load(file)
    # solution = solve(instance_from_scenario_true, solver="run_RHC", is_numerical_exp=True,
    #                  planned_resource_assignment=RHC_info["all_assigned_resources_run"],
    #                  planned_operation_start_times=RHC_info["all_start_times_run"],
    #                  vehicle_original_id=RHC_info["activated_vehicle_id_true"])
    # # visualize_gantt(solution, activated_vehicle_id_true, [], [], None, 'show')
    # visualize_gantt_plotly(solution, RHC_info["activated_vehicle_id_true"], [], [], None, 'show')

    final_solution = RHC(scenario_RHC_config, scenario_exp, scenario_true, False)

    print("Test End")
