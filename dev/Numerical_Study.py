import csv
from Instance import Instance, InstanceConfig
from Solver import solve
from typing import Any, List
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    num_vehicles_pad_buffer_gate_exp: Any = field(default_factory=lambda: [[10, 15, 20, 2, 6],
                                                           [15, 20, 25, 3, 8],
                                                           [20, 25, 30, 4, 10]])
    ETD_margin_exp: List[int] = field(default_factory=lambda: [3, 4, 5])
    ETD_margin_minus_NED_exp: List[int] = field(default_factory=lambda: [1, 2, 3])
    ETA_ready_diff_sigma_exp: List[int] = field(default_factory=lambda: [2, 6, 10])
    is_unified_buffer_exp: List[bool] = field(default_factory=lambda: [True, False])


def Numerical_Experiment(exp_config: ExperimentConfig = ExperimentConfig()):
    csv_file = "Numerical_Experiment_Results.csv"

    with open(csv_file, mode="w", newline="") as f:
        writer = None

        # Run experiments
        for num_vehicles_pad_buffer_gate_exp_list in exp_config.num_vehicles_pad_buffer_gate_exp:
            for num_vehicles in num_vehicles_pad_buffer_gate_exp_list[:3]:
                for ETD_margin in exp_config.ETD_margin_exp:
                    for ETD_margin_minus_NED in exp_config.ETD_margin_minus_NED_exp:
                        for ETA_ready_diff_sigma in exp_config.ETA_ready_diff_sigma_exp:
                            for is_unified_buffer in exp_config.is_unified_buffer_exp:
                                num_solved_prob = 0
                                seed = 0
                                while num_solved_prob < 100:
                                    config = InstanceConfig(num_vehicles=num_vehicles,
                                                            num_pad=num_vehicles_pad_buffer_gate_exp_list[-2],
                                                            num_buffer_in=num_vehicles_pad_buffer_gate_exp_list[-2],
                                                            num_buffer_out=num_vehicles_pad_buffer_gate_exp_list[-2],
                                                            num_gate=num_vehicles_pad_buffer_gate_exp_list[-1],
                                                            ETD_margin=ETD_margin,
                                                            gate_close_margin=ETD_margin - ETD_margin_minus_NED,
                                                            ETA_ready_diff=[3, ETA_ready_diff_sigma],
                                                            is_unified_buffer=is_unified_buffer,
                                                            seed=seed)
                                    instance = Instance.from_config(config)
                                    num_deadlock = 0
                                    results = []
                                    for solver_type in ["FCFS_heuristic", "exact", "FCFS_Gurobi", "FCFS_landing_Gurobi",
                                                        "FCFS_SAT", "FCFS_landing_SAT", "no_rule_SAT"]:
                                        solution = solve(instance, solver=solver_type, is_numerical_exp=True)
                                        if solver_type == "FCFS_heuristic":
                                            solution.solver_runtime = 0.00
                                        num_deadlock += solution.is_deadlock
                                        if solution.is_deadlock:
                                            seed += 1
                                            break
                                        else:
                                            stats = solution.get_summary_stats()
                                            stats.update({
                                                "num_pad": config.num_pad,
                                                "num_buffer_in": config.num_buffer_in,
                                                "num_buffer_out": config.num_buffer_out,
                                                "num_gate": config.num_gate,
                                                "ETD_margin": ETD_margin,
                                                "gate_close_margin": config.gate_close_margin,
                                                "ETA_ready_diff": config.ETA_ready_diff,
                                                "is_unified_buffer": is_unified_buffer,
                                                "seed": seed,
                                                "prob_num": num_solved_prob,
                                            })
                                            results.append(stats)

                                    if num_deadlock == 0:
                                        num_solved_prob += 1
                                        seed += 1

                                        # Initialize writer on first row
                                        if writer is None:
                                            writer = csv.DictWriter(f, fieldnames=results[0].keys())
                                            writer.writeheader()

                                        # Write row immediately after solving
                                        writer.writerows(results)
                                        f.flush()  # ensures row is written to disk
