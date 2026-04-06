# Recovered (best-effort) source from: NumericalStudyRHC.cpython-311.pyc
# Python bytecode version: CPython 3.11
# Bytecode timestamp: 2026-02-24 12:13:03
# Original source path (from .pyc metadata): D:\hoon\Work\[0] Research\[6] Vertiport Operation Scheduling\Algorithm Codes\Vertiport-Surface-Scheduler\dev\NumericalStudyRHC.py

from pathlib import Path
import csv
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Any, List
import time

from Instance import Instance, InstanceConfig
from Solver import solve
from Scenario import ScenarioRHCConfig, Scenario
from RecedingHorizonController import RHC
from Visualizer import visualize_gantt, visualize_gantt_plotly


@dataclass
class ExperimentRHCConfig:
    # NOTE: This field was annotated as `Any` in the bytecode.
    # Each inner list is interpreted as:
    #   [num_vehicles_per_hour_1, ..., num_vehicles_per_hour_k, num_pad, num_gate]
    num_vehicles_pad_buffer_gate_exp: Any = field(
        default_factory=lambda: [[12, 2, 8], [18, 3, 12], [24, 4, 16]]
    )

    # Gate processing-time profiles to test (lists of floats)
    proc_gate_v_exp: List[List[float]] = field(
        default_factory=lambda: [[15.0, 17.0, 23.0, 25.0], [5.0, 6.0, 9.0, 10.0]]
    )

    # Uncertainty levels to test
    uncertainty_level_exp: List[int] = field(default_factory=lambda: [1, 2, 3])

    # Scheduling horizon lengths to test
    scheduling_horizon_length_exp: List[float] = field(default_factory=lambda: [20.0, 30.0, 40.0])

    # Update intervals to test
    update_interval_exp: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])

    # How many repetitions (seeds) per experiment point
    iter: int = 1


def _scenario_filename(exp_config, rhc_config, uncertainty_level, exp_or_true):
    if max(rhc_config.proc_gate_v) == min(max(sub) for sub in exp_config.proc_gate_v_exp):
        return (
            f"scenario_{exp_or_true}"
            f"_v{rhc_config.num_vehicles_per_hour}"
            f"_pad{rhc_config.num_pad}"
            f"_buffer{rhc_config.num_buffer}"
            f"_gate{rhc_config.num_gate}"
            f"_fastcharge_uncertainty{uncertainty_level}"
            f"_horizon{rhc_config.scheduling_horizon_length}"
            f"_update{rhc_config.update_interval}"
            f"_seed{rhc_config.seed}.pkl"
        )
    else:
        return (
            f"scenario_{exp_or_true}"
            f"_v{rhc_config.num_vehicles_per_hour}"
            f"_pad{rhc_config.num_pad}"
            f"_buffer{rhc_config.num_buffer}"
            f"_gate{rhc_config.num_gate}"
            f"_slowcharge_uncertainty{uncertainty_level}"
            f"_horizon{rhc_config.scheduling_horizon_length}"
            f"_update{rhc_config.update_interval}"
            f"_seed{rhc_config.seed}.pkl"
        )


def _scenario_rhc_config_filename(rhc_config, uncertainty_level):
    return (
        f"scenario_RHC_config"
        f"_v{rhc_config.num_vehicles_per_hour}"
        f"_pad{rhc_config.num_pad}"
        f"_buffer{rhc_config.num_buffer}"
        f"_gate{rhc_config.num_gate}"
        f"_uncertainty{uncertainty_level}"
        f"_horizon{rhc_config.scheduling_horizon_length}"
        f"_update{rhc_config.update_interval}"
        f"_seed{rhc_config.seed}.pkl"
    )


def load_scenarios(exp_path, true_path, seed):
    with exp_path.open("rb") as f:
        scenario_exp = pickle.load(f)

    with true_path.open("rb") as f:
        scenario_true = pickle.load(f)

    # Re-seed bridge RNGs (seed + 10000)
    scenario_exp.bridge_rng = np.random.default_rng(seed + 10000)
    scenario_true.bridge_rng = np.random.default_rng(seed + 10000)
    return scenario_exp, scenario_true


def _append_csv_rows(csv_path: Path, rows: list):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)

    with csv_path.open("a", newline="", buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def Numerical_Experiment_RHC(exp_config: ExperimentRHCConfig = ExperimentRHCConfig()):
    csv_path = Path("Numerical_Experiment_RHC_Results.csv")
    scenarios_root = Path("Numerical_Experiment_Scenarios")
    scenarios_root.mkdir(exist_ok=True)

    # Experiment points: [proc_gate_v, uncertainty_level, scheduling_horizon_length, update_interval]
    exp_point_list = [
        [exp_config.proc_gate_v_exp[0], exp_config.uncertainty_level_exp[2],
         exp_config.scheduling_horizon_length_exp[1], exp_config.update_interval_exp[0]],
        [exp_config.proc_gate_v_exp[1], exp_config.uncertainty_level_exp[2],
         exp_config.scheduling_horizon_length_exp[1], exp_config.update_interval_exp[0]],
        [exp_config.proc_gate_v_exp[1], exp_config.uncertainty_level_exp[0],
         exp_config.scheduling_horizon_length_exp[1], exp_config.update_interval_exp[0]],
        [exp_config.proc_gate_v_exp[1], exp_config.uncertainty_level_exp[1],
         exp_config.scheduling_horizon_length_exp[1], exp_config.update_interval_exp[0]],
        [exp_config.proc_gate_v_exp[1], exp_config.uncertainty_level_exp[2],
         exp_config.scheduling_horizon_length_exp[1], exp_config.update_interval_exp[1]],
        [exp_config.proc_gate_v_exp[1], exp_config.uncertainty_level_exp[2],
         exp_config.scheduling_horizon_length_exp[1], exp_config.update_interval_exp[2]],
        [exp_config.proc_gate_v_exp[1], exp_config.uncertainty_level_exp[2],
         exp_config.scheduling_horizon_length_exp[0], exp_config.update_interval_exp[0]],
        [exp_config.proc_gate_v_exp[1], exp_config.uncertainty_level_exp[2],
         exp_config.scheduling_horizon_length_exp[2], exp_config.update_interval_exp[0]],
    ]
    operation_hour = 18

    for cfg_list in exp_config.num_vehicles_pad_buffer_gate_exp:
        pads = cfg_list[-2]
        gates = cfg_list[-1]
        buffers = pads
        vehicles_list = cfg_list[:-2]

        for i in range(len(vehicles_list)):
            num_vehicles_per_hour = vehicles_list[i]
            total_vehicles = num_vehicles_per_hour * operation_hour

            for exp_point in exp_point_list:
                proc_gate_v = exp_point[0]
                uncertainty_level = exp_point[1]
                scheduling_horizon_length = exp_point[2]
                update_interval = exp_point[3]

                # Disturbance range depends on uncertainty level
                if uncertainty_level == 1:
                    disturbance_ready = [-1.0, 5.0, 0.0]
                else:
                    disturbance_ready = [-3.0, 10.0, 0.0]

                for iter in range(exp_config.iter):
                    seed = iter + 42
                    np.random.seed(seed)

                    # Build scenario configuration
                    if uncertainty_level == 3:
                        if num_vehicles_per_hour == 12:
                            scenario_RHC_config = ScenarioRHCConfig(
                                seed=seed,
                                num_vehicles_per_hour=num_vehicles_per_hour,
                                num_pad=pads,
                                num_buffer=buffers,
                                num_gate=gates,
                                is_unified_buffer=True,
                                proc_gate_v=proc_gate_v,
                                disturbance_ready=disturbance_ready,
                                scheduling_horizon_length=scheduling_horizon_length,
                                update_interval=update_interval,
                                operation_hour=operation_hour,
                                dynamic_arrival_v_id=np.random.choice(total_vehicles, size=int(round(total_vehicles * 0.1)), replace=False).tolist(),
                                dynamic_proc_v_id=[99, 117, 78, 8, 35, 150],
                                dynamic_proc_op=[2, 2, 2, 4, 0, 2],
                                dynamic_proc_inc_time=[10.0, 15.0, 10.0, 3.0, 5.0, 20.0],
                            )
                        elif num_vehicles_per_hour == 18:
                            scenario_RHC_config = ScenarioRHCConfig(
                                seed=seed,
                                num_vehicles_per_hour=num_vehicles_per_hour,
                                num_pad=pads,
                                num_buffer=buffers,
                                num_gate=gates,
                                is_unified_buffer=True,
                                proc_gate_v=proc_gate_v,
                                disturbance_ready=disturbance_ready,
                                scheduling_horizon_length=scheduling_horizon_length,
                                update_interval=update_interval,
                                operation_hour=operation_hour,
                                dynamic_arrival_v_id=np.random.choice(total_vehicles, size=int(round(total_vehicles * 0.1)), replace=False).tolist(),
                                dynamic_proc_v_id=[99, 117, 78, 8, 35, 150, 100, 118, 79],
                                dynamic_proc_op=[2, 2, 2, 4, 0, 2, 2, 2, 2],
                                dynamic_proc_inc_time=[10.0, 15.0, 10.0, 3.0, 5.0, 20.0, 10.0, 15.0, 10.0],
                            )
                        elif num_vehicles_per_hour == 24:
                            scenario_RHC_config = ScenarioRHCConfig(
                                seed=seed,
                                num_vehicles_per_hour=num_vehicles_per_hour,
                                num_pad=pads,
                                num_buffer=buffers,
                                num_gate=gates,
                                is_unified_buffer=True,
                                proc_gate_v=proc_gate_v,
                                disturbance_ready=disturbance_ready,
                                scheduling_horizon_length=scheduling_horizon_length,
                                update_interval=update_interval,
                                operation_hour=operation_hour,
                                dynamic_arrival_v_id=np.random.choice(total_vehicles, size=int(round(total_vehicles * 0.1)), replace=False).tolist(),
                                dynamic_proc_v_id=[99, 117, 78, 8, 35, 150, 100, 118, 79, 9, 36, 151],
                                dynamic_proc_op=[2, 2, 2, 4, 0, 2, 2, 2, 2, 4, 0, 2],
                                dynamic_proc_inc_time=[10.0, 15.0, 10.0, 3.0, 5.0, 20.0, 10.0, 15.0, 10.0, 3.0, 5.0, 20.0],
                            )
                    else:
                        scenario_RHC_config = ScenarioRHCConfig(
                            seed=seed,
                            num_vehicles_per_hour=num_vehicles_per_hour,
                            num_pad=pads,
                            num_buffer=buffers,
                            num_gate=gates,
                            is_unified_buffer=True,
                            proc_gate_v=proc_gate_v,
                            disturbance_ready=disturbance_ready,
                            scheduling_horizon_length=scheduling_horizon_length,
                            update_interval=update_interval,
                            operation_hour=operation_hour,
                        )

                    upper_dir = Path("Numerical_Experiment_RHC_Result_Files")
                    upper_dir.mkdir(parents=True, exist_ok=True)
                    if max(scenario_RHC_config.proc_gate_v) == min(max(sub) for sub in exp_config.proc_gate_v_exp):
                        lower_dir = Path(f"v{scenario_RHC_config.num_vehicles_per_hour}"
                                         f"_pad{scenario_RHC_config.num_pad}"
                                         f"_buffer{scenario_RHC_config.num_buffer}"
                                         f"_gate{scenario_RHC_config.num_gate}"
                                         f"_fastcharge_uncertainty{uncertainty_level}"
                                         f"_horizon{scenario_RHC_config.scheduling_horizon_length}"
                                         f"_update{scenario_RHC_config.update_interval}"
                                         f"_seed{scenario_RHC_config.seed}")
                    else:
                        lower_dir = Path(f"v{scenario_RHC_config.num_vehicles_per_hour}"
                                         f"_pad{scenario_RHC_config.num_pad}"
                                         f"_buffer{scenario_RHC_config.num_buffer}"
                                         f"_gate{scenario_RHC_config.num_gate}"
                                         f"_slowcharge_uncertainty{uncertainty_level}"
                                         f"_horizon{scenario_RHC_config.scheduling_horizon_length}"
                                         f"_update{scenario_RHC_config.update_interval}"
                                         f"_seed{scenario_RHC_config.seed}")
                    result_dir = upper_dir / lower_dir
                    result_dir.mkdir(parents=True, exist_ok=True)

                    # Generate scenarios
                    scenario_exp = Scenario.from_scenario_config_exp(scenario_RHC_config)
                    scenario_true = Scenario.from_scenario_config_true(scenario_RHC_config, scenario_exp)

                    scenario_exp.bridge_rng = np.random.default_rng(seed + 10000)
                    scenario_true.bridge_rng = np.random.default_rng(seed + 10000)

                    # Save scenarios
                    scenario_exp_filename = _scenario_filename(exp_config, scenario_RHC_config, uncertainty_level, "exp")
                    scenario_true_filename = _scenario_filename(exp_config, scenario_RHC_config, uncertainty_level, "true")
                    scenario_rhc_config_filename = _scenario_rhc_config_filename(scenario_RHC_config, uncertainty_level)
                    scenario_exp_path = scenarios_root / scenario_exp_filename
                    scenario_true_path = scenarios_root / scenario_true_filename
                    scenario_rhc_config_path = scenarios_root / scenario_rhc_config_filename

                    with scenario_exp_path.open("wb") as file:
                        pickle.dump(scenario_exp, file)
                    with scenario_true_path.open("wb") as file:
                        pickle.dump(scenario_true, file)
                    with scenario_rhc_config_path.open("wb") as file:
                        pickle.dump(scenario_RHC_config, file)

                    results = []

                    # -------------------------
                    # Baseline: FCFS heuristic
                    # -------------------------
                    instance_from_scenario_true, activated_vehicle_id_true = Scenario.scenario_to_instance_true(
                        scenario_true, 1500.0
                    )
                    solution = solve(instance_from_scenario_true, solver="FCFS_heuristic", is_numerical_exp=True)

                    v_id_nominal_proc = list(range(total_vehicles))
                    dynamic_proc_v_id_set = set(scenario_RHC_config.dynamic_proc_v_id)
                    v_id_nominal_proc = [v_id for v_id in v_id_nominal_proc if v_id not in dynamic_proc_v_id_set]
                    vehicle_wise_weighted_tardiness_nominal = solution.vehicle_wise_weighted_tardiness[v_id_nominal_proc]
                    solution.total_weighted_sum_tardiness_nominal = np.sum(vehicle_wise_weighted_tardiness_nominal)
                    solution.max_arrival_tardiness_nominal = np.max(solution.arrival_time_tardiness[v_id_nominal_proc])
                    solution.max_departure_tardiness_nominal = np.max(solution.departure_time_tardiness[v_id_nominal_proc])
                    solution.max_vehicle_wise_weighted_tardiness_nominal = np.max(vehicle_wise_weighted_tardiness_nominal)
                    solution.std_vehicle_wise_weighted_tardiness_nominal = np.std(vehicle_wise_weighted_tardiness_nominal)
                    solution.cv_vehicle_wise_weighted_tardiness_nominal = solution.std_vehicle_wise_weighted_tardiness_nominal / np.mean(vehicle_wise_weighted_tardiness_nominal) if np.mean(vehicle_wise_weighted_tardiness_nominal) > 0 else 0
                    solution.gini_vehicle_wise_weighted_tardiness_nominal = (np.sum(np.abs(vehicle_wise_weighted_tardiness_nominal[:, None] - vehicle_wise_weighted_tardiness_nominal)) / (2 * total_vehicles * np.sum(vehicle_wise_weighted_tardiness_nominal))) if total_vehicles > 0 and np.mean(vehicle_wise_weighted_tardiness_nominal) > 0 else 0

                    path = result_dir / f"FCFS_heuristic_final_solution.pkl"
                    with path.open('wb') as file:
                        pickle.dump(solution, file)
                    visualize_gantt(solution, activated_vehicle_id_true, [], [], None, 'save', result_dir)
                    visualize_gantt_plotly(solution, activated_vehicle_id_true, [], [], None, 'save', None, result_dir)
                    stats = solution.get_summary_stats()

                    if max(proc_gate_v) == min(max(sub) for sub in exp_config.proc_gate_v_exp):
                        stats.update({"charge_type": "Fast"})
                    else:
                        stats.update({"charge_type": "Slow"})

                    stats.update({"objective_option": "None"})
                    stats.update({
                        "num_vehicles_per_hour": num_vehicles_per_hour,
                        "solver": "FCFS_heuristic",
                        "num_pad": pads,
                        "num_buffer_in": buffers,
                        "num_buffer_out": buffers,
                        "num_gate": gates,
                        "uncertainty_level": uncertainty_level,
                        "horizon_length": scheduling_horizon_length,
                        "update_interval": update_interval,
                        "is_unified_buffer": True,
                        "seed": seed,
                        "num_is_runtime_over_true": 0,
                        "num_is_solution_exist_false": 0,
                        "num_is_infeasible_true": 0,
                    })
                    results.append(stats)

                    # --------------------------------
                    # Baseline: randomized FCFS heuristic
                    # --------------------------------
                    scenario_exp, scenario_true = load_scenarios(scenario_exp_path, scenario_true_path, seed)
                    instance_from_scenario_true, activated_vehicle_id_true = Scenario.scenario_to_instance_true(
                        scenario_true, 1500.0
                    )
                    solution = solve(instance_from_scenario_true, solver="FCFS_heuristic_random", is_numerical_exp=True)

                    v_id_nominal_proc = list(range(total_vehicles))
                    dynamic_proc_v_id_set = set(scenario_RHC_config.dynamic_proc_v_id)
                    v_id_nominal_proc = [v_id for v_id in v_id_nominal_proc if v_id not in dynamic_proc_v_id_set]
                    vehicle_wise_weighted_tardiness_nominal = solution.vehicle_wise_weighted_tardiness[v_id_nominal_proc]
                    solution.total_weighted_sum_tardiness_nominal = np.sum(vehicle_wise_weighted_tardiness_nominal)
                    solution.max_arrival_tardiness_nominal = np.max(solution.arrival_time_tardiness[v_id_nominal_proc])
                    solution.max_departure_tardiness_nominal = np.max(solution.departure_time_tardiness[v_id_nominal_proc])
                    solution.max_vehicle_wise_weighted_tardiness_nominal = np.max(vehicle_wise_weighted_tardiness_nominal)
                    solution.std_vehicle_wise_weighted_tardiness_nominal = np.std(vehicle_wise_weighted_tardiness_nominal)
                    solution.cv_vehicle_wise_weighted_tardiness_nominal = solution.std_vehicle_wise_weighted_tardiness_nominal / np.mean(vehicle_wise_weighted_tardiness_nominal) if np.mean(vehicle_wise_weighted_tardiness_nominal) > 0 else 0
                    solution.gini_vehicle_wise_weighted_tardiness_nominal = (np.sum(np.abs(vehicle_wise_weighted_tardiness_nominal[:, None] - vehicle_wise_weighted_tardiness_nominal)) / (2 * total_vehicles * np.sum(vehicle_wise_weighted_tardiness_nominal))) if total_vehicles > 0 and np.mean(vehicle_wise_weighted_tardiness_nominal) > 0 else 0

                    path = result_dir / f"FCFS_heuristic_random_final_solution.pkl"
                    with path.open('wb') as file:
                        pickle.dump(solution, file)
                    visualize_gantt(solution, activated_vehicle_id_true, [], [], None, 'save', result_dir)
                    visualize_gantt_plotly(solution, activated_vehicle_id_true, [], [], None, 'save', None, result_dir)
                    stats = solution.get_summary_stats()

                    if max(proc_gate_v) == min(max(sub) for sub in exp_config.proc_gate_v_exp):
                        stats.update({"charge_type": "Fast"})
                    else:
                        stats.update({"charge_type": "Slow"})

                    stats.update({"objective_option": "None"})
                    stats.update({
                        "num_vehicles_per_hour": num_vehicles_per_hour,
                        "solver": "FCFS_heuristic_random",
                        "num_pad": pads,
                        "num_buffer_in": buffers,
                        "num_buffer_out": buffers,
                        "num_gate": gates,
                        "uncertainty_level": uncertainty_level,
                        "horizon_length": scheduling_horizon_length,
                        "update_interval": update_interval,
                        "is_unified_buffer": True,
                        "seed": seed,
                        "num_is_runtime_over_true": 0,
                        "num_is_solution_exist_false": 0,
                        "num_is_infeasible_true": 0,
                    })
                    results.append(stats)

                    # Write two baseline rows together
                    _append_csv_rows(csv_path, results)

                    # -------------------------
                    # RHC (weighted_sum)
                    # -------------------------
                    solution = None
                    num_is_runtime_over_true = 0
                    num_is_solution_exist_false = 0
                    num_is_infeasible_true = 0

                    scenario_exp, scenario_true = load_scenarios(scenario_exp_path, scenario_true_path, seed)
                    solution, num_is_runtime_over_true, num_is_solution_exist_false, num_is_infeasible_true\
                        = RHC(
                            scenario_RHC_config,
                            scenario_exp,
                            scenario_true,
                            obj_option="weighted_sum",
                            is_file_gen=False,
                            is_result_file_gen=True,
                            scheduler_runtime_limit=10.0,
                            result_dir=result_dir
                        )
                    stats = solution.get_summary_stats()

                    if max(proc_gate_v) == min(max(sub) for sub in exp_config.proc_gate_v_exp):
                        stats.update({"charge_type": "Fast"})
                    else:
                        stats.update({"charge_type": "Slow"})

                    stats.update({
                        "num_vehicles_per_hour": num_vehicles_per_hour,
                        "solver": "RHC",
                        "num_pad": pads,
                        "num_buffer_in": buffers,
                        "num_buffer_out": buffers,
                        "num_gate": gates,
                        "uncertainty_level": uncertainty_level,
                        "horizon_length": scheduling_horizon_length,
                        "update_interval": update_interval,
                        "is_unified_buffer": True,
                        "seed": seed,
                        "num_is_runtime_over_true": num_is_runtime_over_true,
                        "num_is_solution_exist_false": num_is_solution_exist_false,
                        "num_is_infeasible_true": num_is_infeasible_true,
                    })

                    _append_csv_rows(csv_path, [stats])
                    time.sleep(60.0)

                    # -------------------------
                    # RHC (vehicle_wise_max)
                    # -------------------------
                    solution = None
                    num_is_runtime_over_true = 0
                    num_is_solution_exist_false = 0
                    num_is_infeasible_true = 0

                    scenario_exp, scenario_true = load_scenarios(scenario_exp_path, scenario_true_path, seed)
                    solution, num_is_runtime_over_true, num_is_solution_exist_false, num_is_infeasible_true \
                        = RHC(
                            scenario_RHC_config,
                            scenario_exp,
                            scenario_true,
                            obj_option="vehicle_wise_max",
                            is_file_gen=False,
                            is_result_file_gen=True,
                            scheduler_runtime_limit=10.0,
                            result_dir=result_dir
                        )
                    stats = solution.get_summary_stats()

                    if max(proc_gate_v) == min(max(sub) for sub in exp_config.proc_gate_v_exp):
                        stats.update({"charge_type": "Fast"})
                    else:
                        stats.update({"charge_type": "Slow"})

                    stats.update({
                        "num_vehicles_per_hour": num_vehicles_per_hour,
                        "solver": "RHC",
                        "num_pad": pads,
                        "num_buffer_in": buffers,
                        "num_buffer_out": buffers,
                        "num_gate": gates,
                        "uncertainty_level": uncertainty_level,
                        "horizon_length": scheduling_horizon_length,
                        "update_interval": update_interval,
                        "is_unified_buffer": True,
                        "seed": seed,
                        "num_is_runtime_over_true": num_is_runtime_over_true,
                        "num_is_solution_exist_false": num_is_solution_exist_false,
                        "num_is_infeasible_true": num_is_infeasible_true,
                    })

                    _append_csv_rows(csv_path, [stats])
                    time.sleep(60.0)
