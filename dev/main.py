import Solver
import Generator
import Visualizer
import time

if __name__ == "__main__":
    print("This is a module for solving optimization problems using Gurobi.")
    seed, num_ops, num_vehicle, num_pad, num_buffer_in, num_gate, num_buffer_out, num_resource, weights, proc_nominal, proc_width, st_list, ready_max, ETA_ready_diff, ETD_margin = Generator.hyper_param_setting()
    ready, proc, due_a, due_d, ST, vehicle_type, M = Generator.generate_instance(seed, num_ops, num_vehicle, num_pad, num_buffer_in, num_gate, num_buffer_out, proc_nominal, proc_width, st_list, ready_max, ETA_ready_diff, ETD_margin)
    Obj, Gurobi_Runtime, start_time_arr, finish_time_arr, assigned_res_arr, arrival_tar_arr, departure_tar_arr, resource_ind = Solver.solve(seed, num_ops, num_vehicle, num_pad, num_buffer_in, num_gate, num_buffer_out, num_resource, weights, ready, proc, due_a, due_d, ST, M)
    Visualizer.visualize_result(num_ops, num_vehicle, num_pad, num_buffer_in, num_gate, num_buffer_out,
                     proc_nominal, proc_width, st_list, ready_max, ETA_ready_diff, ETD_margin,
                     ready, proc, due_a, due_d, ST, vehicle_type, M,
                     Obj, Gurobi_Runtime, start_time_arr, finish_time_arr, assigned_res_arr,
                     arrival_tar_arr, departure_tar_arr, resource_ind)
