import numpy as np

class Solution:
    def __init__(
        self,
        Obj: float,
        Gurobi_Runtime: float,
        start_time_arr: np.ndarray,
        finish_time_arr: np.ndarray,
        assigned_res_arr: np.ndarray,
        arrival_tar_arr: np.ndarray,
        departure_tar_arr: np.ndarray,
        resource_ind: list,
        solver: str,
    ):
        self.Obj = Obj
        self.Gurobi_Runtime = Gurobi_Runtime
        self.start_time_arr = start_time_arr
        self.finish_time_arr = finish_time_arr
        self.assigned_res_arr = assigned_res_arr
        self.arrival_tar_arr = arrival_tar_arr
        self.departure_tar_arr = departure_tar_arr
        self.resource_ind = resource_ind
        self.solver = solver
