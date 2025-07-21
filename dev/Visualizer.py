import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize_result(num_ops, num_vehicle, num_pad, num_buffer_in, num_gate, num_buffer_out,
                     proc_nominal, proc_width, st_list, ready_max, ETA_ready_diff, ETD_margin,
                     ready, proc, due_a, due_d, ST, vehicle_type, M,
                     Obj, Gurobi_Runtime, start_time_arr, finish_time_arr, assigned_res_arr,
                     arrival_tar_arr, departure_tar_arr, resource_ind):

    num_resource = num_pad + num_buffer_in + num_gate + num_buffer_out
    fig, ax = plt.subplots(figsize=(20, 12))
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
    operation_labels = ['Landing', 'Buffer-In', 'Gate', 'Buffer-Out', 'Take-Off']
    tardiness_color = '#d62728'

    yticks = list(range(num_resource))
    ytick_labels = [f"R{r}" for r in range(num_resource)]

    for v in range(num_vehicle):
        for o in range(num_ops):
            start = start_time_arr[v][o]

            # Determine finish and duration
            if o == 1:
                finish = start_time_arr[v][2]  # Buffer-In ends at Gate start
            elif o == 3:
                finish = start_time_arr[v][4]  # Buffer-Out ends at Takeoff start
            else:
                finish = finish_time_arr[v][o]

            duration = finish - start
            if duration <= 0:
                continue

            res = int(assigned_res_arr[v][o])
            y = res

            # Draw main operation bar
            ax.barh(y, duration, left=start, height=0.4, color=colors[o], edgecolor='black')

            # Vehicle ID centered
            ax.text(start + duration / 2, y, f'V{v}\nP:{duration:.1f}\ntype{vehicle_type[v]}', ha='center', va='center', fontsize=7, color='black')

            # Start time on left edge
            ax.text(start - 0.2, y, f"S:{start:.1f}", ha='left', va='center', fontsize=7, color='black')

            # Finish time on right edge
            ax.text(finish + 0.2, y, f"F:{finish:.1f}", ha='right', va='center', fontsize=7, color='black')

        # Arrival Tardiness (after landing)
        if arrival_tar_arr[v] > 0:
            landing_res = int(assigned_res_arr[v][0])
            landing_end = finish_time_arr[v][0]
            ax.barh(landing_res, arrival_tar_arr[v], left=landing_end - arrival_tar_arr[v],
                    height=0.4, edgecolor=tardiness_color, facecolor='none', hatch='//', linewidth=1.2)

        # Departure Tardiness (after takeoff)
        if departure_tar_arr[v] > 0:
            takeoff_res = int(assigned_res_arr[v][4])
            takeoff_end = finish_time_arr[v][4]
            ax.barh(takeoff_res, departure_tar_arr[v], left=due_d[v],
                    height=0.4, edgecolor=tardiness_color, facecolor='none', hatch='//', linewidth=1.2)

        for v in range(num_vehicle):
            landing_res = int(assigned_res_arr[v][0])
            takeoff_res = int(assigned_res_arr[v][4])
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
                      color='gray', linestyle=':', alpha=0.6)
            ax.text(ready[v], y_landing - 0.3, f"Ready\nV{v}", fontsize=7, color='gray', ha='center')

    # Axis settings
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel('Time (min)')
    ax.set_ylim(-1, num_resource + 1)
    ax.set_title(f"Resource-Centric Gantt | Obj: {Obj:.2f}, "
                 f"Total AT: {sum(arrival_tar_arr):.2f}, Total DT: {sum(departure_tar_arr):.2f}, "
                 f"Runtime: {Gurobi_Runtime:.2f}s")

    # Legend
    legend_ops = [mpatches.Patch(color=colors[i], label=operation_labels[i]) for i in range(num_ops)]
    legend_ops += [
        mpatches.Patch(color=tardiness_color, label='Tardiness', alpha=0.5),
        mpatches.Patch(color='blue', label='ETA', alpha=0.3),
        mpatches.Patch(color='red', label='ETD', alpha=0.3),
        mpatches.Patch(color='gray', label='Ready', alpha=0.3),
        mpatches.Patch(facecolor='white', edgecolor=tardiness_color, hatch='//', label='Tardiness')
    ]
    ax.legend(handles=legend_ops, loc='upper right')

    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
