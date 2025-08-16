import numpy as np
from typing import List, Dict, Optional
from Instance import Instance  # Ensure Instance is imported from its module

class Solution:
    def __init__(
        self,
        # Core solver results
        obj_val: float,
        runtime: float,
        start_times: np.ndarray,
        finish_times: np.ndarray,
        assinged_resources: np.ndarray,
        arrival_time_tardiness: np.ndarray,
        departure_time_tardiness: np.ndarray,
        resource_ind: List[List[int]],
        solver_type: str,
        # Instance data for induced solution informations
        instance: Optional['Instance'] = None,
    ):
        # Core solver results
        self.obj_val = obj_val
        self.solver_type = solver_type
        self.solver_runtime = runtime
        self.start_times = start_times
        self.finish_times = finish_times
        self.assigned_resources = assinged_resources
        self.arrival_time_tardiness = arrival_time_tardiness
        self.departure_time_tardiness = departure_time_tardiness
        self.resource_ind = resource_ind
        # NOTE: refactor to use resource_ind to the mapping of resources to operations ; solver, instance, etc. 
        self.instance = instance

        # Physical characteristics and metadata (from instance)
        if instance:
            self.num_operations = instance.num_operations
            self.num_vehicles = instance.num_vehicles
            self.num_resources = instance.num_resource
            self.num_pad = instance.num_pad
            self.num_buffer_in = instance.num_buffer_in
            self.num_gate = instance.num_gate
            self.num_buffer_out = instance.num_buffer_out
            self.vehicle_type = instance.vehicle_type
            self.ready = instance.vehicle_arrival_times
            self.proc = instance.proc
            self.due_a = instance.vehicle_planed_arrival_times
            self.due_d = instance.vehicle_planed_arrival_times
            self.weights = instance.objective_weights
            self.big_M = instance.big_M
            
            # Computed metrics
            self._compute_derived_metrics()
        else:
            # Fallback for backward compatibility
            self.num_operations = start_times.shape[1] if len(start_times.shape) > 1 else None
            self.num_vehicles = start_times.shape[0] if len(start_times.shape) > 0 else None
            
    def _compute_derived_metrics(self):
        """Compute derived metrics for visualization and analysis"""
        # Ensure we have valid dimensions
        if self.num_vehicles is None or self.num_operations is None:
            return
            
        # Waiting times between operations
        self.waiting_times = np.zeros((self.num_vehicles, self.num_operations - 1))
        for v in range(self.num_vehicles):
            for o in range(self.num_operations - 1):
                next_start = self.start_times[v, o + 1]
                current_finish = self._get_operation_finish_time(v, o)
                self.waiting_times[v, o] = max(0, next_start - current_finish)
        
        # Operation durations
        self.operation_durations = np.zeros((self.num_vehicles, self.num_operations))
        for v in range(self.num_vehicles):
            for o in range(self.num_operations):
                self.operation_durations[v, o] = self._get_operation_duration(v, o)
                
        # Total tardiness metrics
        self.total_arrival_tardiness = np.sum(self.arrival_time_tardiness)
        self.total_departure_tardiness = np.sum(self.departure_time_tardiness)
        
        # Resource utilization
        self.resource_utilization = self._compute_resource_utilization()
        
    def _get_operation_finish_time(self, vehicle: int, operation: int) -> float:
        """Get the actual finish time for an operation considering buffer operations"""
        if self.num_operations is None:
            return self.finish_times[vehicle, operation]
            
        if hasattr(self, 'num_buffer_in') and hasattr(self, 'num_buffer_out'):
            if self.num_buffer_in > 0 and self.num_buffer_out > 0:
                if operation == 1:  # Buffer-In ends at Gate start
                    return self.start_times[vehicle, 2]
                elif operation == 3:  # Buffer-Out ends at Takeoff start
                    return self.start_times[vehicle, self.num_operations - 1]
            elif self.num_buffer_in == 0 and self.num_buffer_out > 0:
                if operation == 2:  # Buffer-Out ends at Takeoff start
                    return self.start_times[vehicle, self.num_operations - 1]
            elif self.num_buffer_in > 0 and self.num_buffer_out == 0:
                if operation == 1:  # Buffer-In ends at Gate start
                    return self.start_times[vehicle, 2]
        
        return self.finish_times[vehicle, operation]
    
    def get_operation_finish_time(self, vehicle: int, operation: int) -> float:
        """Public method to get the actual finish time for an operation considering buffer operations"""
        return self._get_operation_finish_time(vehicle, operation)
    
    def _get_operation_duration(self, vehicle: int, operation: int) -> float:
        """Get the duration of an operation"""
        start = self.start_times[vehicle, operation]
        finish = self._get_operation_finish_time(vehicle, operation)
        return max(0, finish - start)
    
    def _compute_resource_utilization(self) -> Dict[int, float]:
        """Compute utilization percentage for each resource"""
        if not hasattr(self, 'num_resource') or self.num_vehicles is None or self.num_operations is None:
            return {}
            
        utilization = {}
        total_time = np.max(self.finish_times) if self.finish_times.size > 0 else 1
        
        for r in range(self.num_resources):
            busy_time = 0
            for v in range(self.num_vehicles):
                for o in range(self.num_operations):
                    if int(self.assigned_resources[v, o]) == r:
                        busy_time += self._get_operation_duration(v, o)
            utilization[r] = (busy_time / total_time) * 100 if total_time > 0 else 0
            
        return utilization
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics for the solution"""
        stats = {
            'objective_value': self.obj_val,
            'runtime_seconds': self.solver_runtime,
            'total_arrival_tardiness': self.total_arrival_tardiness,
            'total_departure_tardiness': self.total_departure_tardiness,
            'avg_arrival_tardiness': self.total_arrival_tardiness / self.num_vehicles if self.num_vehicles is not None and self.num_vehicles > 0 else 0,
            'avg_departure_tardiness': self.total_departure_tardiness / self.num_vehicles if self.num_vehicles is not None and self.num_vehicles > 0 else 0,
            'total_vehicles': self.num_vehicles if self.num_vehicles is not None else 0,
            'solver': self.solver_type,
        }
        
        if hasattr(self, 'resource_utilization'):
            stats['avg_resource_utilization'] = np.mean(list(self.resource_utilization.values()))
            stats['max_resource_utilization'] = np.max(list(self.resource_utilization.values()))
            
        return stats