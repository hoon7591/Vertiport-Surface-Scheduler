from pathlib import Path
import csv
import pickle
from dataclasses import dataclass, field
from typing import Any, List
from Instance import Instance, InstanceConfig
from Solver import solve


@dataclass
class ExperimentConfig:
    num_vehicles_pad_buffer_gate_exp: Any = field(default_factory=lambda: [
        [10, 15, 2, 8],
        [15, 20, 3, 12],
        [20, 25, 4, 16],
    ])
    proc_gate_v_exp: List[List[float]] = field(default_factory=lambda: [[15.0, 17.0, 23.0, 25.0], [5.0, 6.0, 9.0, 10.0]])
    ETD_margin_exp: List[float] = field(default_factory=lambda: [3.0, 5.0])
    gate_close_margin_exp: List[float] = field(default_factory=lambda: [2.0, 3.0])
    ETA_ready_diff_range_exp: List[List[float]] = field(default_factory=lambda: [[-1.0, 4.0, 3.0], [-3.0, 6.0, 3.0], [-5.0, 8.0, 3.0]])
    is_unified_buffer_exp: List[bool] = field(default_factory=lambda: [True, False])
    ready_max: float = 100.0
    scheduler_runtime_limit: float = 100.0  # in seconds
    iter: int = 100  # number of iteration per case


def _subfolder_name(num_vehicles, pads, buffers, gates, proc_gate_v, ETD_margin, GCM, ETA_ready_diff_range, unified):
    if unified:
        if max(proc_gate_v) <= 10.0:
            return (f"instance_v{num_vehicles}_pad{pads}"
                    f"_buffer{buffers}"
                    f"_gate{gates}_FastCharge_ETD_margin{ETD_margin}"
                    f"_GCM{GCM}_ETA_ready_diff_range{ETA_ready_diff_range[0]}to{ETA_ready_diff_range[1]}at{ETA_ready_diff_range[2]}_unified{unified}")
        else:
            return (f"instance_v{num_vehicles}_pad{pads}"
                    f"_buffer{buffers}"
                    f"_gate{gates}_SlowCharge_ETD_margin{ETD_margin}"
                    f"_GCM{GCM}_ETA_ready_diff_range{ETA_ready_diff_range[0]}to{ETA_ready_diff_range[1]}at{ETA_ready_diff_range[2]}_unified{unified}")
    else:
        if max(proc_gate_v) <= 10.0:
            return (f"instance_v{num_vehicles}_pad{pads}"
                    f"_bufferin{buffers}"
                    f"_bufferout{buffers}"
                    f"_gate{gates}_FastCharge_ETD_margin{ETD_margin}"
                    f"_GCM{GCM}_ETA_ready_diff_range{ETA_ready_diff_range[0]}to{ETA_ready_diff_range[1]}at{ETA_ready_diff_range[2]}_unified{unified}")
        else:
            return (f"instance_v{num_vehicles}_pad{pads}"
                    f"_bufferin{buffers}"
                    f"_bufferout{buffers}"
                    f"_gate{gates}_SlowCharge_ETD_margin{ETD_margin}"
                    f"_GCM{GCM}_ETA_ready_diff_range{ETA_ready_diff_range[0]}to{ETA_ready_diff_range[1]}at{ETA_ready_diff_range[2]}_unified{unified}")


def _instance_filename(num_vehicles, instance, ETD_margin, ETA_ready_diff_range, unified, seed, prob_idx):
    if unified:
        if max(instance.proc_gate_v) <= 10.0:
            return (f"instance_v{num_vehicles}_pad{instance.num_pad}"
                    f"_buffer{instance.num_buffer}"
                    f"_gate{instance.num_gate}_FastCharge_ETD_margin{ETD_margin}"
                    f"_GCM{instance.gate_close_margin}_ETA_ready_diff_range{ETA_ready_diff_range[0]}to{ETA_ready_diff_range[1]}at{ETA_ready_diff_range[2]}"
                    f"_unified{unified}_seed{seed}_prob{prob_idx}.pkl")
        else:
            return (f"instance_v{num_vehicles}_pad{instance.num_pad}"
                    f"_buffer{instance.num_buffer}"
                    f"_gate{instance.num_gate}_SlowCharge_ETD_margin{ETD_margin}"
                    f"_GCM{instance.gate_close_margin}_ETA_ready_diff_range{ETA_ready_diff_range[0]}to{ETA_ready_diff_range[1]}at{ETA_ready_diff_range[2]}"
                    f"_unified{unified}_seed{seed}_prob{prob_idx}.pkl")
    else:
        if max(instance.proc_gate_v) <= 10.0:
            return (f"instance_v{num_vehicles}_pad{instance.num_pad}"
                    f"_bufferin{instance.num_buffer_in}"
                    f"_bufferout{instance.num_buffer_out}"
                    f"_gate{instance.num_gate}_FastCharge_ETD_margin{ETD_margin}"
                    f"_GCM{instance.gate_close_margin}_ETA_ready_diff_range{ETA_ready_diff_range[0]}to{ETA_ready_diff_range[1]}at{ETA_ready_diff_range[2]}"
                    f"_unified{unified}_seed{seed}_prob{prob_idx}.pkl")
        else:
            return (f"instance_v{num_vehicles}_pad{instance.num_pad}"
                    f"_bufferin{instance.num_buffer_in}"
                    f"_bufferout{instance.num_buffer_out}"
                    f"_gate{instance.num_gate}_SlowCharge_ETD_margin{ETD_margin}"
                    f"_GCM{instance.gate_close_margin}_ETA_ready_diff_range{ETA_ready_diff_range[0]}to{ETA_ready_diff_range[1]}at{ETA_ready_diff_range[2]}"
                    f"_unified{unified}_seed{seed}_prob{prob_idx}.pkl")


def _append_csv_rows(csv_path: Path, rows: list):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    with csv_path.open("a", newline="", buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def Numerical_Experiment(exp_config: ExperimentConfig = ExperimentConfig()):
    csv_path = Path("Numerical_Experiment_Results.csv")
    instances_root = Path("Numerical_Experiment_Instances")
    instances_root.mkdir(exist_ok=True)

    # Sweep
    for cfg_list in exp_config.num_vehicles_pad_buffer_gate_exp:
        pads = cfg_list[-2]
        gates = cfg_list[-1]
        buffers = pads  # matches your original choice
        vehicles_list = cfg_list[:-2]

        for i in range(len(vehicles_list)):
            num_vehicles = vehicles_list[i]
            for j in range(len(exp_config.ETD_margin_exp)):
                ETD_margin = exp_config.ETD_margin_exp[j]
                GCM = exp_config.gate_close_margin_exp[j]
                for proc_gate_v in exp_config.proc_gate_v_exp:
                    for ETA_ready_diff_range in exp_config.ETA_ready_diff_range_exp:
                        for unified in exp_config.is_unified_buffer_exp:

                            subfolder = _subfolder_name(
                                num_vehicles, pads, buffers, gates, proc_gate_v,
                                ETD_margin, GCM, ETA_ready_diff_range, unified
                            )
                            out_dir = instances_root / subfolder
                            out_dir.mkdir(parents=True, exist_ok=True)

                            num_solved = 0
                            seed = 0

                            while num_solved < exp_config.iter:
                                # ---- Build instance config (unified vs non-unified) ----
                                if unified:
                                    config = InstanceConfig(
                                        num_vehicles=num_vehicles,
                                        num_pad=pads,
                                        num_buffer=buffers,
                                        num_gate=gates,
                                        proc_gate_v=proc_gate_v,
                                        ETD_margin=ETD_margin,
                                        gate_close_margin=GCM,
                                        ETA_ready_diff=[ETA_ready_diff_range[0], ETA_ready_diff_range[1], ETA_ready_diff_range[2]],
                                        ready_max=exp_config.ready_max,
                                        is_unified_buffer=unified,
                                        seed=seed,
                                    )
                                else:
                                    config = InstanceConfig(
                                        num_vehicles=num_vehicles,
                                        num_pad=pads,
                                        num_buffer_in=buffers,
                                        num_buffer_out=buffers,
                                        num_gate=gates,
                                        proc_gate_v=proc_gate_v,
                                        ETD_margin=ETD_margin,
                                        gate_close_margin=GCM,
                                        ETA_ready_diff=[ETA_ready_diff_range[0], ETA_ready_diff_range[1], ETA_ready_diff_range[2]],
                                        ready_max=exp_config.ready_max,
                                        is_unified_buffer=unified,
                                        seed=seed,
                                    )

                                instance = Instance.from_config(config)

                                # ---- Save instance pickle ----
                                pkl_name = _instance_filename(
                                    num_vehicles, instance, ETD_margin, ETA_ready_diff_range, unified, seed, num_solved
                                )
                                pkl_path = out_dir / pkl_name
                                with pkl_path.open("wb") as fh:
                                    pickle.dump(instance, fh)

                                # ---- Run solvers ----
                                num_deadlock = 0
                                num_runtime_over = 0
                                results = []

                                # for solver_type in ["FCFS_heuristic", "FCFS_heuristic_random", "exact", "FCFS_Gurobi",    # whole solver options (7 cases)
                                #                     "FCFS_landing_Gurobi", "FCFS_SAT", "FCFS_landing_SAT", "no_rule_SAT"]:
                                for solver_type in ["FCFS_heuristic", "FCFS_heuristic_random", "exact"]:
                                    solution = solve(instance, solver=solver_type, is_numerical_exp=True,
                                                     scheduler_runtime_limit=exp_config.scheduler_runtime_limit)
                                    num_deadlock += int(getattr(solution, "is_deadlock", False))
                                    num_runtime_over += int(getattr(solution, "is_runtime_over", False))

                                    if solution.is_deadlock or solution.is_runtime_over:
                                        # discard this instance; try next seed
                                        seed += 1
                                        try:
                                            pkl_path.unlink(missing_ok=True)
                                        finally:
                                            pass
                                        break
                                    else:
                                        # write one row per successful solver immediately
                                        stats = solution.get_summary_stats()
                                        if config.is_unified_buffer:
                                            if max(proc_gate_v) <= 10.0:
                                                stats.update({"charge_type": "Fast"})
                                            else:
                                                stats.update({"charge_type": "Slow"})
                                            if solver_type != "exact":
                                                stats.update({"objective_option": "None"})
                                            stats.update({
                                                "solver": solver_type,  # keep explicit
                                                "num_pad": config.num_pad,
                                                "num_buffer_in": config.num_buffer,
                                                "num_buffer_out": config.num_buffer,
                                                "num_gate": config.num_gate,
                                                "ETD_margin": ETD_margin,
                                                "gate_close_margin": config.gate_close_margin,
                                                "ETA_ready_diff": config.ETA_ready_diff,
                                                "is_unified_buffer": unified,
                                                "seed": seed,
                                                "prob_num": num_solved,
                                            })
                                        else:
                                            if max(proc_gate_v) <= 10.0:
                                                stats.update({"charge_type": "Fast"})
                                            else:
                                                stats.update({"charge_type": "Slow"})
                                            if solver_type != "exact":
                                                stats.update({"objective_option": "None"})
                                            stats.update({
                                                "solver": solver_type,  # keep explicit
                                                "num_pad": config.num_pad,
                                                "num_buffer_in": config.num_buffer_in,
                                                "num_buffer_out": config.num_buffer_out,
                                                "num_gate": config.num_gate,
                                                "ETD_margin": ETD_margin,
                                                "gate_close_margin": config.gate_close_margin,
                                                "ETA_ready_diff": config.ETA_ready_diff,
                                                "is_unified_buffer": unified,
                                                "seed": seed,
                                                "prob_num": num_solved,
                                            })
                                        results.append(stats)

                                        if solver_type == "exact":
                                            solution = solve(instance, solver=solver_type, is_numerical_exp=True,
                                                             scheduler_runtime_limit=exp_config.scheduler_runtime_limit,
                                                             obj_option="vehicle_wise_max")
                                            num_deadlock += int(getattr(solution, "is_deadlock", False))
                                            num_runtime_over += int(getattr(solution, "is_runtime_over", False))

                                            if solution.is_deadlock or solution.is_runtime_over:
                                                # discard this instance; try next seed
                                                seed += 1
                                                try:
                                                    pkl_path.unlink(missing_ok=True)
                                                finally:
                                                    pass
                                                break
                                            else:
                                                # write one row per successful solver immediately
                                                stats = solution.get_summary_stats()
                                                if config.is_unified_buffer:
                                                    if max(proc_gate_v) <= 10.0:
                                                        stats.update({"charge_type": "Fast"})
                                                    else:
                                                        stats.update({"charge_type": "Slow"})
                                                    stats.update({
                                                        "solver": solver_type,  # keep explicit
                                                        "num_pad": config.num_pad,
                                                        "num_buffer_in": config.num_buffer,
                                                        "num_buffer_out": config.num_buffer,
                                                        "num_gate": config.num_gate,
                                                        "ETD_margin": ETD_margin,
                                                        "gate_close_margin": config.gate_close_margin,
                                                        "ETA_ready_diff": config.ETA_ready_diff,
                                                        "is_unified_buffer": unified,
                                                        "seed": seed,
                                                        "prob_num": num_solved,
                                                    })
                                                else:
                                                    if max(proc_gate_v) <= 10.0:
                                                        stats.update({"charge_type": "Fast"})
                                                    else:
                                                        stats.update({"charge_type": "Slow"})
                                                    stats.update({
                                                        "solver": solver_type,  # keep explicit
                                                        "num_pad": config.num_pad,
                                                        "num_buffer_in": config.num_buffer_in,
                                                        "num_buffer_out": config.num_buffer_out,
                                                        "num_gate": config.num_gate,
                                                        "ETD_margin": ETD_margin,
                                                        "gate_close_margin": config.gate_close_margin,
                                                        "ETA_ready_diff": config.ETA_ready_diff,
                                                        "is_unified_buffer": unified,
                                                        "seed": seed,
                                                        "prob_num": num_solved,
                                                    })
                                                results.append(stats)

                                # ---- Count only if both solvers passed ----
                                if num_deadlock == 0 and num_runtime_over == 0:
                                    _append_csv_rows(csv_path, results)
                                    num_solved += 1
                                    seed += 1
