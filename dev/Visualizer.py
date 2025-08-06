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
    num_ops = solution.num_ops
    num_resource = solution.num_resource
    num_vehicle = solution.num_vehicle
    num_pad = solution.num_pad
    num_buffer_in = solution.num_buffer_in
    num_gate = solution.num_gate
    num_buffer_out = solution.num_buffer_out
    
    # Core solution data
    Obj = solution.Obj
    Gurobi_Runtime = solution.Gurobi_Runtime
    start_time_arr = solution.start_time_arr
    finish_time_arr = solution.finish_time_arr
    assigned_res_arr = solution.assigned_res_arr
    arrival_tar_arr = solution.arrival_tar_arr
    departure_tar_arr = solution.departure_tar_arr
    resource_ind = solution.resource_ind
    solver = solution.solver
    
    # Physical characteristics from solution
    vehicle_type = solution.vehicle_type
    ready = solution.ready
    due_a = solution.due_a
    due_d = solution.due_d
    
    # Generate visualization info
    operation_labels, colors = _get_operation_display_info(num_buffer_in, num_buffer_out)
    tardiness_color = '#d62728'

    fig, ax = plt.subplots(figsize=(20, 12))

    yticks = list(range(num_resource))
    ytick_labels = [f"R{r}" for r in range(num_resource)]

    for v in range(num_vehicle):
        for o in range(num_ops):
            start = start_time_arr[v][o]
            
            # Use solution's method to get finish time
            finish = solution._get_operation_finish_time(v, o)
            duration = solution.operation_durations[v, o]
            
            if duration <= 0:
                continue

            res = int(assigned_res_arr[v][o])
            y = res

            # Draw main operation bar
            ax.barh(y, duration, left=start, height=0.4, color=colors[o], edgecolor='black', zorder=1)

            # Draw waiting bar using solution's waiting times
            if o != num_ops - 1:
                waiting_duration = solution.waiting_times[v, o]
                if waiting_duration > 0:
                    ax.barh(y, waiting_duration, left=finish, height=0.4, color='gray', edgecolor='black', zorder=1)
                    ax.text(start_time_arr[v][o + 1], y, f'{start_time_arr[v][o + 1]:.1f}', ha='left', va='center', fontsize=7, color='black')

            # Vehicle ID centered
            ax.text(start + duration / 2, y, f'V{v}\nP:{duration:.1f}\ntype{vehicle_type[v]}', ha='center', va='center', fontsize=7, color='black')

            # Start time on left edge
            ax.text(start - 0.2, y, f"S:{start:.1f}", ha='left', va='center', fontsize=7, color='black')

            # Finish time on right edge
            ax.text(finish + 0.2, y, f"F:{finish:.1f}", ha='right', va='center', fontsize=7, color='black')

        # Arrival Tardiness (after landing)
        if arrival_tar_arr[v] > 0:
            landing_res = int(assigned_res_arr[v][0])
            landing_end = solution._get_operation_finish_time(v, 0)
            ax.barh(landing_res, arrival_tar_arr[v], left=landing_end - arrival_tar_arr[v],
                    height=0.4, edgecolor=tardiness_color, facecolor='none', hatch='//', linewidth=1.2, zorder=2)

        # Departure Tardiness (after takeoff)
        if departure_tar_arr[v] > 0:
            takeoff_res = int(assigned_res_arr[v][num_ops - 1])
            takeoff_end = solution._get_operation_finish_time(v, num_ops - 1)
            ax.barh(takeoff_res, departure_tar_arr[v], left=due_d[v],
                    height=0.4, edgecolor=tardiness_color, facecolor='none', hatch='//', linewidth=1.2, zorder=2)

    for v in range(num_vehicle):
        landing_res = int(assigned_res_arr[v][0])
        takeoff_res = int(assigned_res_arr[v][num_ops - 1])
        y_landing = landing_res
        y_takeoff = takeoff_res

        # ETA marker on Landing row
        ax.vlines(due_a[v], ymin=y_landing - 0.2, ymax=y_landing + 0.2,
                  color='blue', linestyle='--', alpha=0.6)
        ax.text(due_a[v], y_landing + 0.3, f"ETA\nV{v}\nAT:{arrival_tar_arr[v]:.1f}", fontsize=7, color='blue', ha='center')

        # ETD marker on Takeoff row
        ax.vlines(due_d[v], ymin=y_takeoff - 0.2, ymax=y_takeoff + 0.2,
                  color='red', linestyle='--', alpha=0.6)
        ax.text(due_d[v], y_takeoff - 0.42, f"ETD\nV{v}\nDT:{departure_tar_arr[v]:.1f}", fontsize=7, color='red', ha='center')

        # Ready marker on Landing row
        ax.vlines(ready[v], ymin=y_landing - 0.2, ymax=y_landing + 0.2,
                  color='darkmagenta', linestyle=':', alpha=0.6)
        ax.text(ready[v], y_landing - 0.3, f"Ready\nV{v}", fontsize=7, color='darkmagenta', ha='center')

    # Axis settings
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel('Time (min)')
    ax.set_ylim(-1, num_resource + 1)
    ax.set_title(f"Resource-Centric Gantt | Obj: {Obj:.2f}, "
                 f"Total AT: {solution.total_arrival_tardiness:.2f}, Total DT: {solution.total_departure_tardiness:.2f}, "
                 f"Runtime: {Gurobi_Runtime:.2f}s, Solver: {solver}")

    # Legend
    legend_ops = [mpatches.Patch(color=colors[i], label=operation_labels[i]) for i in range(num_ops)]
    legend_ops += [
        mpatches.Patch(color='gray', label='Waiting', alpha=0.5),
        mpatches.Patch(facecolor='white', edgecolor=tardiness_color, hatch='//', label='Tardiness')
    ]
    ax.legend(handles=legend_ops, loc='upper right')

    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
