import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import plotly.graph_objects as go


def _get_operation_display_info(is_unified_buffer, num_buffer_in, num_buffer_out, num_buffer):
    """Get operation labels and colors for visualization based on buffer configuration"""
    if is_unified_buffer:
        if num_buffer == 0:
            colors = ['#66c2a5', '#8da0cb', '#a6d854']
            operation_labels = ['Landing', 'Gate', 'Take-Off']
        else:
            colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
            operation_labels = ['Landing', 'Buffer-In', 'Gate', 'Buffer-Out', 'Take-Off']
    else:
        if num_buffer_in == 0:
            if num_buffer_out == 0:
                colors = ['#66c2a5', '#8da0cb', '#a6d854']
                operation_labels = ['Landing', 'Gate', 'Take-Off']
            else:
                colors = ['#66c2a5', '#8da0cb', '#e78ac3', '#a6d854']
                operation_labels = ['Landing', 'Gate', 'Buffer-Out', 'Take-Off']
        else:
            if num_buffer_out == 0:
                colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#a6d854']
                operation_labels = ['Landing', 'Buffer-In', 'Gate', 'Take-Off']
            else:
                colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
                operation_labels = ['Landing', 'Buffer-In', 'Gate', 'Buffer-Out', 'Take-Off']
            
    return operation_labels, colors


def visualize_gantt(solution, vehicle_ids, *arg):
    """
    arg[0]: List of Processing Vehicles IDs to Skip for Visualization
    arg[1]: List of Processing Vehicles Operation IDs to Skip for Visualization
    arg[2]: Filename Suffix for Saving the Figure (Current Time)
    arg[3]: Visualization Mode ('show' or 'save')
    """

    # Use solution's enhanced data instead of extracting from instance
    num_operations = solution.num_operations
    num_resources = solution.num_resources
    num_vehicles = solution.num_vehicles
    num_buffer_in = solution.num_buffer_in
    num_buffer_out = solution.num_buffer_out
    num_buffer = solution.num_buffer
    is_unified_buffer = solution.is_unified_buffer
    
    # Core solution data
    obj_val = solution.obj_val
    solver_runtime = solution.solver_runtime
    sim_end_time = solution.sim_end_time
    start_times = solution.start_times
    assigned_resources = solution.assigned_resources
    arrival_tardiness = solution.arrival_time_tardiness
    departure_tardiness = solution.departure_time_tardiness
    solver_type = solution.solver_type
    
    # Physical characteristics from solution
    vehicle_type = solution.vehicle_type
    ready = solution.ready
    planned_arrival_times = solution.vehicle_planned_arrival_times
    planned_departure_times = solution.vehicle_planned_departure_times
    planned_gate_closing_times = solution.vehicle_planned_gate_close_times
    
    # Generate visualization info
    operation_labels, colors = _get_operation_display_info(is_unified_buffer, num_buffer_in, num_buffer_out, num_buffer)
    tardiness_color = '#d62728'

    _, ax = plt.subplots(figsize=(20, 12))

    yticks = list(range(num_resources))
    ytick_labels = [f"R{r}" for r in range(num_resources)]

    tardiness_correction = 0.0

    for v in range(len(vehicle_ids)):
        for o in range(num_operations):
            if len(arg) > 0 and vehicle_ids[v] in arg[0] and o < arg[1][arg[0].index(vehicle_ids[v])]:
                continue
            else:
                start = start_times[v][o]

                # Use solution's method to get finish time
                finish = solution.get_operation_finish_time(v, o)
                duration = solution.operation_durations[v, o]
                waiting_duration = 0.0
                if o != num_operations - 1:
                    waiting_duration = solution.waiting_times[v, o]

                if duration <= 1e-06 and waiting_duration <= 1e-06:
                    continue

                res = int(assigned_resources[v][o])
                y = res

                # Draw main operation bar
                if duration > 1e-06:
                    ax.barh(y, duration, left=start, height=0.4, color=colors[o], edgecolor='black', zorder=1)

                # Draw waiting bar using solution's waiting times
                if o != num_operations - 1:
                    if waiting_duration > 1e-06:
                        ax.barh(y, waiting_duration, left=finish, height=0.4, color='gray', edgecolor='black', zorder=1)
                        ax.text(start_times[v][o + 1], y, f'{start_times[v][o + 1]:.1f}', ha='left', va='center', fontsize=10, color='black')

                # Vehicle ID centered
                if duration > 1e-06:
                    ax.text(start + duration / 2, y, f'V{vehicle_ids[v]}\nP:{duration:.1f}\ntype{vehicle_type[v]}', ha='center', va='center', fontsize=10, color='black')

                    # Start time on left edge
                    ax.text(start - 0.2, y, f"S:{start:.1f}", ha='left', va='center', fontsize=10, color='black')

                    # Finish time on right edge
                    ax.text(finish + 0.2, y, f"F:{finish:.1f}", ha='right', va='center', fontsize=10, color='black')

        if len(arg) > 0 and vehicle_ids[v] in arg[0] and arg[1][arg[0].index(vehicle_ids[v])] == 4:
            tardiness_correction += departure_tardiness[v]
            continue
        else:
            # Departure Tardiness (after takeoff)
            if departure_tardiness[v] > 0:
                takeoff_resource_idx = int(assigned_resources[v][num_operations - 1])
                ax.barh(takeoff_resource_idx, departure_tardiness[v], left=planned_departure_times[v],
                        height=0.4, edgecolor=tardiness_color, facecolor='none', hatch='//', linewidth=1.2, zorder=2)

        if len(arg) > 0 and vehicle_ids[v] in arg[0] and arg[1][arg[0].index(vehicle_ids[v])] > 0:
            tardiness_correction += arrival_tardiness[v]
            continue
        else:
            # Arrival Tardiness (after landing)
            if arrival_tardiness[v] > 0:
                landing_resource_idx = int(assigned_resources[v][0])
                ax.barh(landing_resource_idx, arrival_tardiness[v], left=planned_arrival_times[v],
                        height=0.4, edgecolor=tardiness_color, facecolor='none', hatch='//', linewidth=1.2, zorder=2)

    for v in range(len(vehicle_ids)):
        landing_resource_idx = int(assigned_resources[v][0])
        if is_unified_buffer:
            if num_buffer > 0:
                gate_resource_idx = int(assigned_resources[v][num_operations - 3])
            else:
                gate_resource_idx = int(assigned_resources[v][num_operations - 2])
        else:
            if num_buffer_out > 0:
                gate_resource_idx = int(assigned_resources[v][num_operations - 3])
            else:
                gate_resource_idx = int(assigned_resources[v][num_operations - 2])
        takeoff_resource_idx = int(assigned_resources[v][num_operations - 1])
        y_landing = landing_resource_idx
        y_gate = gate_resource_idx
        y_takeoff = takeoff_resource_idx

        if len(arg) > 0 and vehicle_ids[v] in arg[0] and arg[1][arg[0].index(vehicle_ids[v])] == 4:
            continue
        else:
            # ETD marker on Takeoff row
            ax.vlines(planned_departure_times[v], ymin=y_takeoff - 0.2, ymax=y_takeoff + 0.2,
                      color='red', linestyle='--', alpha=0.6)
            ax.text(planned_departure_times[v], y_takeoff - 0.75,
                    f"ETD\nV{vehicle_ids[v]}\nDT:{departure_tardiness[v]:.1f}", fontsize=10, color='red', ha='center')

        if len(arg) > 0 and vehicle_ids[v] in arg[0] and arg[1][arg[0].index(vehicle_ids[v])] > 2:
            continue
        else:
            # Gate closing time marker on Gate row (no early departure)
            ax.vlines(planned_gate_closing_times[v], ymin=y_gate - 0.2, ymax=y_gate + 0.2,
                      color='green', linestyle='--', alpha=0.6)
            ax.text(planned_gate_closing_times[v], y_gate - 0.5, f"NED\nV{vehicle_ids[v]}", fontsize=10, color='green',
                    ha='center')

        if len(arg) > 0 and vehicle_ids[v] in arg[0] and arg[1][arg[0].index(vehicle_ids[v])] > 0:
            continue
        else:
            # ETA marker on Landing row
            ax.vlines(planned_arrival_times[v], ymin=y_landing - 0.2, ymax=y_landing + 0.2,
                      color='blue', linestyle='--', alpha=0.6)
            ax.text(planned_arrival_times[v], y_landing + 0.25, f"ETA\nV{vehicle_ids[v]}\nAT:{arrival_tardiness[v]:.1f}", fontsize=10, color='blue', ha='center')

        if len(arg) > 0 and vehicle_ids[v] in arg[0]:
            continue
        else:
            # Ready marker on Landing row
            ax.vlines(ready[v], ymin=y_landing - 0.2, ymax=y_landing + 0.2,
                      color='darkmagenta', linestyle=':', alpha=0.6)
            ax.text(ready[v], y_landing - 0.5, f"Ready\nV{vehicle_ids[v]}", fontsize=10, color='darkmagenta', ha='center')

    # Axis settings
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel('Time (min)')
    ax.set_ylim(-1, num_resources + 1)
    if solver_type in ["exact", "exact_RHC", "FCFS_SAT", "FCFS_Gurobi", "FCFS_landing_SAT", "FCFS_landing_Gurobi"]:
        ax.set_title(f"Resource-Centric Gantt | Obj: {obj_val - tardiness_correction:.2f}, "
                     f"Total AT: {solution.total_arrival_tardiness:.2f}min, Total DT: {solution.total_departure_tardiness:.2f}min, "
                     f"Runtime: {solver_runtime:.2f}s, Sim End Time: {sim_end_time:.2f}min, "
                     f"Solver: {solver_type}, Obj Option: {solution.objective_option}\n"
                     f"Total T: {solution.total_arrival_tardiness + solution.total_departure_tardiness:.2f}min, "
                     f"Average T: {(solution.total_arrival_tardiness + solution.total_departure_tardiness) / num_vehicles if num_vehicles > 0 else 0:.2f}min, "
                     f"Max AT: {solution.max_arrival_tardiness:.2f}min, Max DT: {solution.max_departure_tardiness:.2f}min, "
                     f"Max Vehicle-wise T: {solution.max_vehicle_wise_tardiness:.2f}min, Num Vehicles: {num_vehicles}, "
                     f"Weights: [{solution.objective_weights[0]:.2f}, {solution.objective_weights[1]:.2f}]")
    else:
        ax.set_title(f"Resource-Centric Gantt | Obj: {obj_val - tardiness_correction:.2f}, "
                     f"Total AT: {solution.total_arrival_tardiness:.2f}min, Total DT: {solution.total_departure_tardiness:.2f}min, "
                     f"Runtime: {solver_runtime:.2f}s, Sim End Time: {sim_end_time:.2f}min, Solver: {solver_type}\n"
                     f"Total T: {solution.total_arrival_tardiness + solution.total_departure_tardiness:.2f}min, "
                     f"Average T: {(solution.total_arrival_tardiness + solution.total_departure_tardiness) / num_vehicles if num_vehicles > 0 else 0:.2f}min, "
                     f"Max AT: {solution.max_arrival_tardiness:.2f}min, Max DT: {solution.max_departure_tardiness:.2f}min, "
                     f"Max Vehicle-wise T: {solution.max_vehicle_wise_tardiness:.2f}min, Num Vehicles: {num_vehicles}, "
                     f"Weights: [{solution.objective_weights[0]:.2f}, {solution.objective_weights[1]:.2f}]")

    # Legend
    legend_ops = [mpatches.Patch(color=colors[i], label=operation_labels[i]) for i in range(num_operations)]
    legend_ops += [
        mpatches.Patch(color='gray', label='Waiting', alpha=0.5),
        mpatches.Patch(facecolor='white', edgecolor=tardiness_color, hatch='//', label='Tardiness')
    ]
    ax.legend(handles=legend_ops, loc='upper right')

    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if arg[3] == 'show':
        plt.show()
    elif arg[3] == 'save':
        plt.savefig(f'{solver_type}_{arg[2]}.pdf', dpi=300)
        plt.close()
        print(f"Saved interactive Gantt to {solver_type}_{arg[2]}.pdf")


def visualize_gantt_plotly(
    solution,
    vehicle_ids,
    skip_vehicle_ids=None,
    skip_vehicle_ops=None,
    filename_suffix="0.0",
    mode="show",              # "show" or "save"
    current_time=None         # optional vertical line for RHC time
):
    """
    Plot an interactive resource-centric Gantt chart using Plotly.

    Parameters
    ----------
    solution : Solution
        The solution object containing schedule info.
    vehicle_ids : list[int]
        Indices (or IDs) of vehicles to visualize (index aligned with solution arrays).
    skip_vehicle_ids : list[int] or None
        Vehicles for which part of operations / markers are skipped.
    skip_vehicle_ops : list[int] or None
        For each vehicle in skip_vehicle_ids, integer specifying how many operations to skip in visualization logic
        (kept consistent with your original Matplotlib logic).
    filename_suffix : str
        Suffix to add in output filename when mode == "save".
    mode : {"show", "save"}
        "show" -> fig.show(), "save" -> write_html(...)
    current_time : float or None
        If given, draw a vertical line at this time (e.g., RHC horizon start).
    """

    if skip_vehicle_ids is None:
        skip_vehicle_ids = []
    if skip_vehicle_ops is None:
        skip_vehicle_ops = []

    # --- Core solution info ---
    num_operations = solution.num_operations
    num_resources = solution.num_resources
    num_vehicles = solution.num_vehicles
    num_buffer_in = solution.num_buffer_in
    num_buffer_out = solution.num_buffer_out
    num_buffer = solution.num_buffer
    is_unified_buffer = solution.is_unified_buffer

    obj_val = solution.obj_val
    solver_runtime = solution.solver_runtime
    sim_end_time = solution.sim_end_time
    start_times = solution.start_times
    assigned_resources = solution.assigned_resources
    arrival_tardiness = solution.arrival_time_tardiness
    departure_tardiness = solution.departure_time_tardiness
    solver_type = solution.solver_type

    vehicle_type = solution.vehicle_type
    ready = solution.ready
    planned_arrival_times = solution.vehicle_planned_arrival_times
    planned_departure_times = solution.vehicle_planned_departure_times
    planned_gate_closing_times = solution.vehicle_planned_gate_close_times

    operation_labels, colors = _get_operation_display_info(
        is_unified_buffer, num_buffer_in, num_buffer_out, num_buffer
    )

    tardiness_color = "rgba(214, 39, 40, 1.0)"  # red

    # y-axis labels: one row per resource
    ytick_labels = [f"R{r}" for r in range(num_resources)]

    fig = go.Figure()

    # To avoid legend spam
    op_legend_shown = [False] * num_operations
    waiting_legend_shown = False
    arr_tard_legend_shown = False
    dep_tard_legend_shown = False
    eta_legend_shown = False
    etd_legend_shown = False
    ned_legend_shown = False
    ready_legend_shown = False

    tardiness_correction = 0.0

    # Helper to check skip info
    def get_skip_op_for_vehicle(veh_id):
        if veh_id in skip_vehicle_ids:
            idx = skip_vehicle_ids.index(veh_id)
            return skip_vehicle_ops[idx]
        return None

    # --- 1. Operation and waiting bars ---
    for v_idx, veh_id in enumerate(vehicle_ids):
        skip_level = get_skip_op_for_vehicle(veh_id)

        for o in range(num_operations):
            if skip_level is not None and o < skip_level:
                # Same semantics as your original "skip" logic
                continue

            start = float(start_times[v_idx, o])
            finish = float(solution.get_operation_finish_time(v_idx, o))
            duration = float(solution.operation_durations[v_idx, o])

            waiting_duration = 0.0
            if o != num_operations - 1:
                waiting_duration = float(solution.waiting_times[v_idx, o])

            if duration <= 1e-6 and waiting_duration <= 1e-6:
                continue

            res = int(assigned_resources[v_idx, o])
            y_label = f"R{res}"

            # Operation bar
            if duration > 1e-6:
                fig.add_trace(go.Bar(
                    x=[duration],
                    y=[y_label],
                    base=[start],
                    orientation="h",
                    marker=dict(color=colors[o]),
                    name=operation_labels[o],
                    showlegend=not op_legend_shown[o],
                    hovertemplate=(
                        f"<b>Vehicle V{veh_id}</b><br>"
                        f"Op: {operation_labels[o]} (#{o})<br>"
                        f"Resource: {y_label}<br>"
                        f"Start: {start:.1f}<br>"
                        f"Finish: {finish:.1f}<br>"
                        f"Duration: {duration:.1f}<br>"
                        f"Type: {vehicle_type[v_idx]}<extra></extra>"
                    )
                ))
                op_legend_shown[o] = True

            # Waiting bar (after op o)
            if o != num_operations - 1 and waiting_duration > 1e-6:
                fig.add_trace(go.Bar(
                    x=[waiting_duration],
                    y=[y_label],
                    base=[finish],
                    orientation="h",
                    marker=dict(color="rgba(120, 120, 120, 0.4)"),
                    name="Waiting",
                    showlegend=not waiting_legend_shown,
                    hovertemplate=(
                        f"<b>Vehicle V{veh_id}</b><br>"
                        f"Resource: {y_label}<br>"
                        f"Waiting: {finish:.1f} → {finish + waiting_duration:.1f}<extra></extra>"
                    )
                ))
                waiting_legend_shown = True

    # --- 2. Tardiness bands (arrival / departure) ---
    for v_idx, veh_id in enumerate(vehicle_ids):
        skip_level = get_skip_op_for_vehicle(veh_id)

        # Departure tardiness
        if skip_level == 4:
            tardiness_correction += departure_tardiness[v_idx]
        else:
            if departure_tardiness[v_idx] > 1e-6:
                takeoff_res = int(assigned_resources[v_idx, num_operations - 1])
                y_label = f"R{takeoff_res}"
                fig.add_trace(go.Bar(
                    x=[float(departure_tardiness[v_idx])],
                    y=[y_label],
                    base=[float(planned_departure_times[v_idx])],
                    orientation="h",
                    marker=dict(
                        color="rgba(214, 39, 40, 0.18)",
                        line=dict(color=tardiness_color, width=1.0)
                    ),
                    name="Departure Tardiness",
                    showlegend=not dep_tard_legend_shown,
                    hovertemplate=(
                        f"<b>Vehicle V{veh_id}</b><br>"
                        f"Resource: {y_label}<br>"
                        f"Departure tardiness: {departure_tardiness[v_idx]:.1f} min<br>"
                        f"Planned dep: {planned_departure_times[v_idx]:.1f}<extra></extra>"
                    )
                ))
                dep_tard_legend_shown = True

        # Arrival tardiness
        if skip_level is not None and skip_level > 0:
            tardiness_correction += arrival_tardiness[v_idx]
        else:
            if arrival_tardiness[v_idx] > 1e-6:
                landing_res = int(assigned_resources[v_idx, 0])
                y_label = f"R{landing_res}"
                fig.add_trace(go.Bar(
                    x=[float(arrival_tardiness[v_idx])],
                    y=[y_label],
                    base=[float(planned_arrival_times[v_idx])],
                    orientation="h",
                    marker=dict(
                        color="rgba(214, 39, 40, 0.18)",
                        line=dict(color=tardiness_color, width=1.0)
                    ),
                    name="Arrival Tardiness",
                    showlegend=not arr_tard_legend_shown,
                    hovertemplate=(
                        f"<b>Vehicle V{veh_id}</b><br>"
                        f"Resource: {y_label}<br>"
                        f"Arrival tardiness: {arrival_tardiness[v_idx]:.1f} min<br>"
                        f"Planned arr: {planned_arrival_times[v_idx]:.1f}<extra></extra>"
                    )
                ))
                arr_tard_legend_shown = True

    # --- 3. Markers: ETA, NED (gate close), ETD, Ready ---
    for v_idx, veh_id in enumerate(vehicle_ids):
        skip_level = get_skip_op_for_vehicle(veh_id)

        landing_res = int(assigned_resources[v_idx, 0])
        takeoff_res = int(assigned_resources[v_idx, num_operations - 1])

        if is_unified_buffer:
            if num_buffer > 0:
                gate_res = int(assigned_resources[v_idx, num_operations - 3])
            else:
                gate_res = int(assigned_resources[v_idx, num_operations - 2])
        else:
            if num_buffer_out > 0:
                gate_res = int(assigned_resources[v_idx, num_operations - 3])
            else:
                gate_res = int(assigned_resources[v_idx, num_operations - 2])

        y_landing = f"R{landing_res}"
        y_gate = f"R{gate_res}"
        y_takeoff = f"R{takeoff_res}"

        # ETD marker
        if not (skip_level == 4):
            fig.add_trace(go.Scatter(
                x=[float(planned_departure_times[v_idx])],
                y=[y_takeoff],
                mode="markers+text",
                marker=dict(color="red", symbol="triangle-up", size=10),
                text=[f"ETD<br>V{veh_id}"],
                textposition="top center",
                name="ETD",
                showlegend=not etd_legend_shown,
                hovertemplate=(
                    f"<b>Vehicle V{veh_id}</b><br>"
                    f"ETD: {planned_departure_times[v_idx]:.1f}<br>"
                    f"DT: {departure_tardiness[v_idx]:.1f} min<extra></extra>"
                )
            ))
            etd_legend_shown = True

        # Gate closing (NED)
        if not (skip_level is not None and skip_level > 2):
            fig.add_trace(go.Scatter(
                x=[float(planned_gate_closing_times[v_idx])],
                y=[y_gate],
                mode="markers+text",
                marker=dict(color="green", symbol="x", size=10),
                text=[f"NED<br>V{veh_id}"],
                textposition="bottom center",
                name="Gate Close (NED)",
                showlegend=not ned_legend_shown,
                hovertemplate=(
                    f"<b>Vehicle V{veh_id}</b><br>"
                    f"Gate close: {planned_gate_closing_times[v_idx]:.1f}<extra></extra>"
                )
            ))
            ned_legend_shown = True

        # ETA marker
        if not (skip_level is not None and skip_level > 0):
            fig.add_trace(go.Scatter(
                x=[float(planned_arrival_times[v_idx])],
                y=[y_landing],
                mode="markers+text",
                marker=dict(color="blue", symbol="triangle-down", size=10),
                text=[f"ETA<br>V{veh_id}"],
                textposition="top center",
                name="ETA",
                showlegend=not eta_legend_shown,
                hovertemplate=(
                    f"<b>Vehicle V{veh_id}</b><br>"
                    f"ETA: {planned_arrival_times[v_idx]:.1f}<br>"
                    f"AT: {arrival_tardiness[v_idx]:.1f} min<extra></extra>"
                )
            ))
            eta_legend_shown = True

        # Ready marker
        if skip_level is None:
            fig.add_trace(go.Scatter(
                x=[float(ready[v_idx])],
                y=[y_landing],
                mode="markers+text",
                marker=dict(color="magenta", symbol="circle-open", size=10),
                text=[f"Ready<br>V{veh_id}"],
                textposition="bottom center",
                name="Ready",
                showlegend=not ready_legend_shown,
                hovertemplate=(
                    f"<b>Vehicle V{veh_id}</b><br>"
                    f"Ready: {ready[v_idx]:.1f}<extra></extra>"
                )
            ))
            ready_legend_shown = True

    # --- 4. Optional current time vertical line (for RHC) ---
    if current_time is not None:
        fig.add_vline(
            x=current_time,
            line_color="black",
            line_dash="dash",
            annotation_text=f"t = {current_time:.1f}",
            annotation_position="top left"
        )

    # --- 5. Layout & title ---
    if solver_type in ["exact", "exact_RHC", "FCFS_SAT", "FCFS_Gurobi", "FCFS_landing_SAT", "FCFS_landing_Gurobi"]:
        title_str = (
            f"Resource-Centric Gantt | Obj: {obj_val - tardiness_correction:.2f}, "
            f"Total AT: {solution.total_arrival_tardiness:.2f} min, Total DT: {solution.total_departure_tardiness:.2f} min, "
            f"Runtime: {solver_runtime:.2f} s, Sim End: {sim_end_time:.2f} min, "
            f"Solver: {solver_type}, Obj: {solution.objective_option}<br>"
            f"Total T: {solution.total_arrival_tardiness + solution.total_departure_tardiness:.2f} min, "
            f"Avg T: {(solution.total_arrival_tardiness + solution.total_departure_tardiness) / num_vehicles if num_vehicles > 0 else 0:.2f} min, "
            f"Max AT: {solution.max_arrival_tardiness:.2f} min, Max DT: {solution.max_departure_tardiness:.2f} min, "
            f"Max Vehicle-wise T: {solution.max_vehicle_wise_tardiness:.2f} min, Num V: {num_vehicles}, "
            f"W = [{solution.objective_weights[0]:.2f}, {solution.objective_weights[1]:.2f}]"
        )
    else:
        title_str = (
            f"Resource-Centric Gantt | Obj: {obj_val - tardiness_correction:.2f}, "
            f"Total AT: {solution.total_arrival_tardiness:.2f} min, Total DT: {solution.total_departure_tardiness:.2f} min, "
            f"Runtime: {solver_runtime:.2f} s, Sim End: {sim_end_time:.2f} min, Solver: {solver_type}<br>"
            f"Total T: {solution.total_arrival_tardiness + solution.total_departure_tardiness:.2f} min, "
            f"Avg T: {(solution.total_arrival_tardiness + solution.total_departure_tardiness) / num_vehicles if num_vehicles > 0 else 0:.2f} min, "
            f"Max AT: {solution.max_arrival_tardiness:.2f} min, Max DT: {solution.max_departure_tardiness:.2f} min, "
            f"Max Vehicle-wise T: {solution.max_vehicle_wise_tardiness:.2f} min, Num V: {num_vehicles}, "
            f"W = [{solution.objective_weights[0]:.2f}, {solution.objective_weights[1]:.2f}]"
        )

    fig.update_yaxes(
        title_text="Resource",
        categoryorder="array",
        categoryarray=ytick_labels
    )
    fig.update_xaxes(title_text="Time (min)")

    fig.update_layout(
        title=title_str,
        barmode="overlay",
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0
        ),
        template="plotly_white"
    )

    # --- 6. Show or save ---
    if mode == "show":
        fig.show()
    elif mode == "save":
        out_name = f"{solution.solver_type}_{filename_suffix}.html"
        fig.write_html(out_name)
        print(f"Saved interactive Gantt to {out_name}")


def visualize_top5_vehicle_wise_delay(solution, vehicle_ids):
    # Define info data
    num_vehicles = solution.num_vehicles
    arrival_tardiness = solution.arrival_time_tardiness
    departure_tardiness = solution.departure_time_tardiness
    total_tardiness = arrival_tardiness + departure_tardiness

    # Extract Top 5 vehicles by total tardiness
    top5_tardy_vehicle_ids = np.argsort(total_tardiness)[-5:][::-1]
    top5_arrival_tardiness = arrival_tardiness[top5_tardy_vehicle_ids]
    top5_departure_tardiness = departure_tardiness[top5_tardy_vehicle_ids]
    top5_total_tardiness = total_tardiness[top5_tardy_vehicle_ids]

    sorted_order = np.argsort(top5_total_tardiness)
    top5_tardy_vehicle_ids = top5_tardy_vehicle_ids[sorted_order]
    top5_arrival_tardiness = top5_arrival_tardiness[sorted_order]
    top5_departure_tardiness = top5_departure_tardiness[sorted_order]
    top5_total_tardiness = top5_total_tardiness[sorted_order]

    # Draw bars and text for tardiness chart
    _, ax = plt.subplots(figsize=(12, 8))

    yticks = list(range(len(top5_tardy_vehicle_ids)))
    ytick_labels = [f"V{v}" for v in top5_tardy_vehicle_ids]

    for i in range(len(top5_tardy_vehicle_ids)):
        ax.barh(i, top5_arrival_tardiness[i], left=0.0, height=0.5, color='red', edgecolor='black', zorder=1)
        ax.barh(i, top5_departure_tardiness[i], left=top5_arrival_tardiness[i], height=0.5, color='blue', edgecolor='black', zorder=1)
        if top5_arrival_tardiness[i] <= 0.2:
            ax.text(0.01, i - 0.27, f'AT:{top5_arrival_tardiness[i]:.2f}', ha='left', va='top', fontsize=10, color='black')
        else:
            ax.text(top5_arrival_tardiness[i] / 2, i, f'AT:{top5_arrival_tardiness[i]:.2f}', ha='center', va='center', fontsize=10, color='black')
        if top5_departure_tardiness[i] <= 0.2:
            ax.text(top5_arrival_tardiness[i] + 0.01, i - 0.27, f'DT:{top5_departure_tardiness[i]:.2f}', ha='left', va='top', fontsize=10, color='black')
        else:
            ax.text(top5_arrival_tardiness[i] + top5_departure_tardiness[i] / 2, i, f'DT:{top5_departure_tardiness[i]:.2f}', ha='center', va='center', fontsize=10, color='black')
        ax.text(top5_arrival_tardiness[i] + top5_departure_tardiness[i] + 0.15, i, f'Total:{top5_total_tardiness[i]:.2f}', ha='left', va='center', fontsize=10, color='black')

    ax.set_xlim(0.0, np.max(total_tardiness) * 1.2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    plt.show()
