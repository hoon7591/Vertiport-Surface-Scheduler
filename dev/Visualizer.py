import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _get_operation_display_info(num_buffer_in, num_buffer_out):
    """Get operation labels and colors for visualization based on buffer configuration"""
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


def visualize_result(solution):
    # Use solution's enhanced data instead of extracting from instance
    num_operations = solution.num_operations
    num_resources = solution.num_resources
    num_vehicles = solution.num_vehicles
    num_buffer_in = solution.num_buffer_in
    num_buffer_out = solution.num_buffer_out
    
    # Core solution data
    obj_val = solution.obj_val
    solver_runtime = solution.solver_runtime
    start_times = solution.start_times
    assigned_resources = solution.assigned_resources
    arrival_time_tardiness = solution.arrival_time_tardiness
    departure_time_tardiness = solution.departure_time_tardiness
    solver_type = solution.solver_type
    
    # Physical characteristics from solution
    vehicle_type = solution.vehicle_type
    ready = solution.ready
    due_a = solution.due_a
    due_d = solution.due_d
    
    # Generate visualization info
    operation_labels, colors = _get_operation_display_info(num_buffer_in, num_buffer_out)
    tardiness_color = '#d62728'

    _, ax = plt.subplots(figsize=(20, 12))

    yticks = list(range(num_resources))
    ytick_labels = [f"R{r}" for r in range(num_resources)]

    for v in range(num_vehicles):
        for o in range(num_operations):
            start = start_times[v][o]
            
            # Use solution's method to get finish time
            finish = solution.get_operation_finish_time(v, o)
            duration = solution.operation_durations[v, o]
            
            if duration <= 0:
                continue

            res = int(assigned_resources[v][o])
            y = res

            # Draw main operation bar
            ax.barh(y, duration, left=start, height=0.4, color=colors[o], edgecolor='black', zorder=1)

            # Draw waiting bar using solution's waiting times
            if o != num_operations - 1:
                waiting_duration = solution.waiting_times[v, o]
                if waiting_duration > 0:
                    ax.barh(y, waiting_duration, left=finish, height=0.4, color='gray', edgecolor='black', zorder=1)
                    ax.text(start_times[v][o + 1], y, f'{start_times[v][o + 1]:.1f}', ha='left', va='center', fontsize=7, color='black')

            # Vehicle ID centered
            ax.text(start + duration / 2, y, f'V{v}\nP:{duration:.1f}\ntype{vehicle_type[v]}', ha='center', va='center', fontsize=7, color='black')

            # Start time on left edge
            ax.text(start - 0.2, y, f"S:{start:.1f}", ha='left', va='center', fontsize=7, color='black')

            # Finish time on right edge
            ax.text(finish + 0.2, y, f"F:{finish:.1f}", ha='right', va='center', fontsize=7, color='black')

        # Arrival Tardiness (after landing)
        if arrival_time_tardiness[v] > 0:
            landing_res = int(assigned_resources[v][0])
            landing_end = solution.get_operation_finish_time(v, 0)
            ax.barh(landing_res, arrival_time_tardiness[v], left=landing_end - arrival_time_tardiness[v],
                    height=0.4, edgecolor=tardiness_color, facecolor='none', hatch='//', linewidth=1.2, zorder=2)

        # Departure Tardiness (after takeoff)
        if departure_time_tardiness[v] > 0:
            takeoff_res = int(assigned_resources[v][num_operations - 1])
            ax.barh(takeoff_res, departure_time_tardiness[v], left=due_d[v],
                    height=0.4, edgecolor=tardiness_color, facecolor='none', hatch='//', linewidth=1.2, zorder=2)

    for v in range(num_vehicles):
        landing_res = int(assigned_resources[v][0])
        takeoff_res = int(assigned_resources[v][num_operations - 1])
        y_landing = landing_res
        y_takeoff = takeoff_res

        # ETA marker on Landing row
        ax.vlines(due_a[v], ymin=y_landing - 0.2, ymax=y_landing + 0.2,
                  color='blue', linestyle='--', alpha=0.6)
        ax.text(due_a[v], y_landing + 0.3, f"ETA\nV{v}\nAT:{arrival_time_tardiness[v]:.1f}", fontsize=7, color='blue', ha='center')

        # ETD marker on Takeoff row
        ax.vlines(due_d[v], ymin=y_takeoff - 0.2, ymax=y_takeoff + 0.2,
                  color='red', linestyle='--', alpha=0.6)
        ax.text(due_d[v], y_takeoff - 0.42, f"ETD\nV{v}\nDT:{departure_time_tardiness[v]:.1f}", fontsize=7, color='red', ha='center')

        # Ready marker on Landing row
        ax.vlines(ready[v], ymin=y_landing - 0.2, ymax=y_landing + 0.2,
                  color='darkmagenta', linestyle=':', alpha=0.6)
        ax.text(ready[v], y_landing - 0.3, f"Ready\nV{v}", fontsize=7, color='darkmagenta', ha='center')

    # Axis settings
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel('Time (min)')
    ax.set_ylim(-1, num_resources + 1)
    ax.set_title(f"Resource-Centric Gantt | Obj: {obj_val:.2f}, "
                 f"Total AT: {solution.total_arrival_tardiness:.2f}, Total DT: {solution.total_departure_tardiness:.2f}, "
                 f"Runtime: {solver_runtime:.2f}s, Solver: {solver_type}")

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
