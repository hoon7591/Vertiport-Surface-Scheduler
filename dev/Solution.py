import numpy as np
from typing import List, Dict, Tuple, Optional
from Instance import Instance  # Ensure Instance is imported from its module

class Solution:
    def __init__(
        self,
        # Core solver results
        Obj: float,
        Gurobi_Runtime: float,
        start_time_arr: np.ndarray,
        finish_time_arr: np.ndarray,
        assigned_res_arr: np.ndarray,
        arrival_tar_arr: np.ndarray,
        departure_tar_arr: np.ndarray,
        resource_ind: List[List[int]],
        solver: str,
        # Instance data for physical characteristics
        instance: Optional['Instance'] = None,
    ):
        # Core solver results
        self.Obj = Obj
        self.Gurobi_Runtime = Gurobi_Runtime
        self.start_time_arr = start_time_arr
        self.finish_time_arr = finish_time_arr
        self.assigned_res_arr = assigned_res_arr
        self.arrival_tar_arr = arrival_tar_arr
        self.departure_tar_arr = departure_tar_arr
        self.resource_ind = resource_ind
        self.solver = solver
        
        # Physical characteristics and metadata (from instance)
        if instance:
            self.num_ops = instance.num_ops
            self.num_vehicle = instance.num_vehicle
            self.num_resource = instance.num_resource
            self.num_pad = instance.num_pad
            self.num_buffer_in = instance.num_buffer_in
            self.num_gate = instance.num_gate
            self.num_buffer_out = instance.num_buffer_out
            self.vehicle_type = instance.vehicle_type
            self.ready = instance.ready
            self.proc = instance.proc
            self.due_a = instance.due_a
            self.due_d = instance.due_d
            self.weights = instance.weights
            self.M = instance.M
            
            # Computed metrics
            self._compute_derived_metrics()
        else:
            # Fallback for backward compatibility
            self.num_ops = start_time_arr.shape[1] if len(start_time_arr.shape) > 1 else None
            self.num_vehicle = start_time_arr.shape[0] if len(start_time_arr.shape) > 0 else None
            
    def _compute_derived_metrics(self):
        """Compute derived metrics for visualization and analysis"""
        # Waiting times between operations
        self.waiting_times = np.zeros((self.num_vehicle, self.num_ops - 1))
        for v in range(self.num_vehicle):
            for o in range(self.num_ops - 1):
                next_start = self.start_time_arr[v, o + 1]
                current_finish = self._get_operation_finish_time(v, o)
                self.waiting_times[v, o] = max(0, next_start - current_finish)
        
        # Operation durations
        self.operation_durations = np.zeros((self.num_vehicle, self.num_ops))
        for v in range(self.num_vehicle):
            for o in range(self.num_ops):
                self.operation_durations[v, o] = self._get_operation_duration(v, o)
                
        # Total tardiness metrics
        self.total_arrival_tardiness = np.sum(self.arrival_tar_arr)
        self.total_departure_tardiness = np.sum(self.departure_tar_arr)
        
        # Resource utilization
        self.resource_utilization = self._compute_resource_utilization()
        
    def _get_operation_finish_time(self, vehicle: int, operation: int) -> float:
        """Get the actual finish time for an operation considering buffer operations"""
        if hasattr(self, 'num_buffer_in') and hasattr(self, 'num_buffer_out'):
            if self.num_buffer_in > 0 and self.num_buffer_out > 0:
                if operation == 1:  # Buffer-In ends at Gate start
                    return self.start_time_arr[vehicle, 2]
                elif operation == 3:  # Buffer-Out ends at Takeoff start
                    return self.start_time_arr[vehicle, self.num_ops - 1]
            elif self.num_buffer_in == 0 and self.num_buffer_out > 0:
                if operation == 2:  # Buffer-Out ends at Takeoff start
                    return self.start_time_arr[vehicle, self.num_ops - 1]
            elif self.num_buffer_in > 0 and self.num_buffer_out == 0:
                if operation == 1:  # Buffer-In ends at Gate start
                    return self.start_time_arr[vehicle, 2]
        
        return self.finish_time_arr[vehicle, operation]
    
    def _get_operation_duration(self, vehicle: int, operation: int) -> float:
        """Get the duration of an operation"""
        start = self.start_time_arr[vehicle, operation]
        finish = self._get_operation_finish_time(vehicle, operation)
        return max(0, finish - start)
    
    def _compute_resource_utilization(self) -> Dict[int, float]:
        """Compute utilization percentage for each resource"""
        if not hasattr(self, 'num_resource'):
            return {}
            
        utilization = {}
        total_time = np.max(self.finish_time_arr) if self.finish_time_arr.size > 0 else 1
        
        for r in range(self.num_resource):
            busy_time = 0
            for v in range(self.num_vehicle):
                for o in range(self.num_ops):
                    if int(self.assigned_res_arr[v, o]) == r:
                        busy_time += self._get_operation_duration(v, o)
            utilization[r] = (busy_time / total_time) * 100 if total_time > 0 else 0
            
        return utilization
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics for the solution"""
        stats = {
            'objective_value': self.Obj,
            'runtime_seconds': self.Gurobi_Runtime,
            'total_arrival_tardiness': self.total_arrival_tardiness,
            'total_departure_tardiness': self.total_departure_tardiness,
            'avg_arrival_tardiness': self.total_arrival_tardiness / self.num_vehicle if self.num_vehicle > 0 else 0,
            'avg_departure_tardiness': self.total_departure_tardiness / self.num_vehicle if self.num_vehicle > 0 else 0,
            'total_vehicles': self.num_vehicle,
            'solver': self.solver,
        }
        
        if hasattr(self, 'resource_utilization'):
            stats['avg_resource_utilization'] = np.mean(list(self.resource_utilization.values()))
            stats['max_resource_utilization'] = np.max(list(self.resource_utilization.values()))
            
        return stats
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics for the solution"""
        stats = {
            'objective_value': self.Obj,
            'runtime_seconds': self.Gurobi_Runtime,
            'total_arrival_tardiness': self.total_arrival_tardiness,
            'total_departure_tardiness': self.total_departure_tardiness,
            'avg_arrival_tardiness': self.total_arrival_tardiness / self.num_vehicle if self.num_vehicle > 0 else 0,
            'avg_departure_tardiness': self.total_departure_tardiness / self.num_vehicle if self.num_vehicle > 0 else 0,
            'total_vehicles': self.num_vehicle,
            'solver': self.solver,
        }
        
        if hasattr(self, 'resource_utilization'):
            stats['avg_resource_utilization'] = np.mean(list(self.resource_utilization.values()))
            stats['max_resource_utilization'] = np.max(list(self.resource_utilization.values()))
            
        return stats
