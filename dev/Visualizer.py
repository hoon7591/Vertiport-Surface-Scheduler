import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


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
    plt.show()


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
