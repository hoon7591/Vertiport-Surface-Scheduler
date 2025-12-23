import copy
import numpy as np
from typing import Dict, List, Any, Optional
from Instance import Instance  # Ensure Instance is imported from its module

class Solution:
    def __init__(
        self,
        # Core solver results
        obj_val: float,
        runtime: float,
        sim_end_time: float,
        start_times: np.ndarray,
        finish_times: np.ndarray,
        assinged_resources: np.ndarray,
        arrival_time_tardiness: np.ndarray,
        departure_time_tardiness: np.ndarray,
        resource_ind: List[List[int]],
        solver_type: str,
        objective_weights: List[float],
        # Instance data for induced solution informations
        instance: Optional['Instance'] = None,
        is_deadlock: bool = False,
        is_runtime_over: bool = False,
        objective_option: Optional[str] = None,
    ):
        # Core solver results
        self.obj_val = obj_val
        self.solver_type = solver_type
        self.solver_runtime = runtime
        self.sim_end_time = sim_end_time
        self.start_times = start_times
        self.finish_times = finish_times
        self.assigned_resources = assinged_resources
        self.arrival_time_tardiness = arrival_time_tardiness
        self.departure_time_tardiness = departure_time_tardiness
        self.resource_ind = resource_ind
        # NOTE: refactor to use resource_ind to the mapping of resources to operations ; solver, instance, etc. 
        self.instance = instance
        self.is_deadlock = is_deadlock
        self.is_runtime_over = is_runtime_over
        self.objective_option = objective_option
        self.objective_weights = objective_weights

        # Physical characteristics and metadata (from instance)
        if instance:
            self.num_operations = instance.num_operations
            self.num_vehicles = instance.num_vehicles
            self.num_resources = instance.num_resource
            self.num_pad = instance.num_pad
            self.num_buffer_in = instance.num_buffer_in
            self.num_gate = instance.num_gate
            self.num_buffer_out = instance.num_buffer_out
            self.num_buffer = instance.num_buffer
            self.is_unified_buffer = instance.is_unified_buffer
            self.vehicle_type = instance.vehicle_type
            self.ready = instance.vehicle_arrival_times
            self.proc = instance.proc
            self.vehicle_planned_arrival_times = instance.vehicle_planned_arrival_times
            self.vehicle_planned_departure_times = instance.vehicle_planned_departure_times
            self.vehicle_planned_gate_close_times = instance.vehicle_planned_gate_close_times
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

        # Operation durations
        self.operation_durations = np.zeros((self.num_vehicles, self.num_operations))
        for v in range(self.num_vehicles):
            for o in range(self.num_operations):
                self.operation_durations[v, o] = self._get_operation_duration(v, o)
        self.operation_durations_consider_buffer = copy.deepcopy(self.operation_durations)

        # Waiting times between operations
        self.waiting_times = np.zeros((self.num_vehicles, self.num_operations - 1))
        self.waiting_times_consider_buffer = np.zeros((self.num_vehicles, self.num_operations - 1))
        for v in range(self.num_vehicles):
            for o in range(self.num_operations - 1):
                next_start = self.start_times[v, o + 1]
                current_finish = self._get_operation_finish_time(v, o)
                wait = max(0.0, next_start - current_finish)
                self.waiting_times[v, o] = wait
                self.waiting_times_consider_buffer[v, o] = wait
        buffer_ops = []
        if self.is_unified_buffer:
            if self.num_buffer > 0:
                buffer_ops = [self.num_operations - 4, self.num_operations - 2]
        else:
            if self.num_buffer_in > 0 and self.num_buffer_out > 0:
                buffer_ops = [self.num_operations - 4, self.num_operations - 2]
            elif self.num_buffer_in > 0:
                buffer_ops = [self.num_operations - 3]
            elif self.num_buffer_out > 0:
                buffer_ops = [self.num_operations - 2]
        for v in range(self.num_vehicles):
            for o in buffer_ops:
                self.waiting_times_consider_buffer[v, o] = self.operation_durations[v, o]
                self.operation_durations_consider_buffer[v, o] = 0.0

        # Constrained and unconstrained waiting times between operations
        self.constrained_waiting_times = np.zeros((self.num_vehicles, self.num_operations - 1))
        self.unconstrained_waiting_times = copy.deepcopy(self.waiting_times_consider_buffer)
        for v in range(self.num_vehicles):
            planned_gate_close = self.vehicle_planned_gate_close_times[v]
            if self.is_unified_buffer:
                if self.num_buffer > 0:
                    current_finish = self._get_operation_finish_time(v, self.num_operations - 3)
                    self.constrained_waiting_times[v, self.num_operations - 3] = max(0, planned_gate_close - current_finish)
                    self.unconstrained_waiting_times[v, self.num_operations - 3] -= self.constrained_waiting_times[v, self.num_operations - 3]
                else:
                    current_finish = self._get_operation_finish_time(v, self.num_operations - 2)
                    self.constrained_waiting_times[v, self.num_operations - 2] = max(0, planned_gate_close - current_finish)
                    self.unconstrained_waiting_times[v, self.num_operations - 2] -= self.constrained_waiting_times[v, self.num_operations - 2]
            else:
                if self.num_buffer_out > 0:
                    current_finish = self._get_operation_finish_time(v, self.num_operations - 3)
                    self.constrained_waiting_times[v, self.num_operations - 3] = max(0, planned_gate_close - current_finish)
                    self.unconstrained_waiting_times[v, self.num_operations - 3] -= self.constrained_waiting_times[v, self.num_operations - 3]
                else:
                    current_finish = self._get_operation_finish_time(v, self.num_operations - 2)
                    self.constrained_waiting_times[v, self.num_operations - 2] = max(0, planned_gate_close - current_finish)
                    self.unconstrained_waiting_times[v, self.num_operations - 2] -= self.constrained_waiting_times[v, self.num_operations - 2]
                
        # Total tardiness metrics
        self.total_arrival_tardiness = np.sum(self.arrival_time_tardiness)
        self.total_departure_tardiness = np.sum(self.departure_time_tardiness)
        self.total_tardiness = self.total_arrival_tardiness + self.total_departure_tardiness
        self.avg_arrival_tardiness = self.total_arrival_tardiness / self.num_vehicles if self.num_vehicles > 0 else 0
        self.avg_departure_tardiness = self.total_departure_tardiness / self.num_vehicles if self.num_vehicles > 0 else 0
        self.avg_total_tardiness = self.total_tardiness / self.num_vehicles if self.num_vehicles > 0 else 0

        # Max tardiness metrics
        self.max_arrival_tardiness = np.max(self.arrival_time_tardiness)
        self.max_departure_tardiness = np.max(self.departure_time_tardiness)
        vehicle_wise_tardiness = self.arrival_time_tardiness + self.departure_time_tardiness
        self.max_vehicle_wise_tardiness = np.max(vehicle_wise_tardiness)

        # Fairness of tardiness metrics
        self.std_vehicle_wise_tardiness = np.std(vehicle_wise_tardiness)
        non_delayed_vehicles_mask = np.isclose(vehicle_wise_tardiness, 0.0, atol=1e-9)
        self.non_delayed_vehicles_rate = np.sum(non_delayed_vehicles_mask) / self.num_vehicles if self.num_vehicles > 0 else 0
        
        # Resource utilization
        self.total_idle_time = np.zeros(self.num_resources)
        self.total_processing_time = np.zeros(self.num_resources)
        self.total_waiting_time = np.zeros(self.num_resources)
        self.total_constrained_waiting_time = np.zeros(self.num_resources)
        self.total_unconstrained_waiting_time = np.zeros(self.num_resources)
        self.total_occupation_time = np.zeros(self.num_resources)
        self.resource_occupation_rate = np.zeros(self.num_resources)
        self.effective_occupation_rate = np.zeros(self.num_resources)
        self.ineffective_occupation_rate = np.zeros(self.num_resources)
        self.total_time = np.max(self.finish_times) if self.finish_times.size > 0 else 1

        self.assigned_resources_int = np.rint(self.assigned_resources).astype(int)
        for v in range(self.num_vehicles):
            for o in range(self.num_operations):
                self.total_processing_time[self.assigned_resources_int[v, o]] += self.operation_durations_consider_buffer[v, o]
                self.total_waiting_time[self.assigned_resources_int[v, o]] += self.waiting_times_consider_buffer[v, o] if o < self.num_operations - 1 else 0
                self.total_constrained_waiting_time[self.assigned_resources_int[v, o]] += self.constrained_waiting_times[v, o] if o < self.num_operations - 1 else 0
                self.total_unconstrained_waiting_time[self.assigned_resources_int[v, o]] += self.unconstrained_waiting_times[v, o] if o < self.num_operations - 1 else 0

        for r in range(self.num_resources):
            self.total_idle_time[r] = self.total_time - (self.total_processing_time[r] + self.total_waiting_time[r])
            self.total_occupation_time[r] = self.total_processing_time[r] + self.total_waiting_time[r]
            self.resource_occupation_rate[r] = (self.total_occupation_time[r] / self.total_time) * 100 if self.total_time > 0 else 0
            self.effective_occupation_rate[r] = (self.total_processing_time[r] / self.total_occupation_time[r]) * 100 if self.total_occupation_time[r] > 0 else 0
            self.ineffective_occupation_rate[r] = (self.total_unconstrained_waiting_time[r] / self.total_occupation_time[r]) * 100 if self.total_occupation_time[r] > 0 else 0

        self.pad_resource_occupation_rate = np.mean(self.resource_occupation_rate[:self.num_pad]) if self.num_pad > 0 else 0
        self.pad_effective_occupation_rate = np.mean(self.effective_occupation_rate[:self.num_pad]) if self.num_pad > 0 else 0
        self.pad_ineffective_occupation_rate = np.mean(self.ineffective_occupation_rate[:self.num_pad]) if self.num_pad > 0 else 0
        self.gate_resource_occupation_rate = np.mean(self.resource_occupation_rate[-self.num_gate:]) if self.num_gate > 0 else 0
        self.gate_effective_occupation_rate = np.mean(self.effective_occupation_rate[-self.num_gate:]) if self.num_gate > 0 else 0
        self.gate_ineffective_occupation_rate = np.mean(self.ineffective_occupation_rate[-self.num_gate:]) if self.num_gate > 0 else 0
        self.total_resource_occupation_rate = np.mean(self.resource_occupation_rate) if self.num_resources > 0 else 0
        self.total_effective_occupation_rate = np.mean(self.effective_occupation_rate) if self.num_resources > 0 else 0
        self.total_ineffective_occupation_rate = np.mean(self.ineffective_occupation_rate) if self.num_resources > 0 else 0
        self.pad_unconstrained_waiting_time = np.sum(self.total_unconstrained_waiting_time[:self.num_pad]) if self.num_pad > 0 else 0
        self.gate_unconstrained_waiting_time = np.sum(self.total_unconstrained_waiting_time[-self.num_gate:]) if self.num_gate > 0 else 0
        self.buffer_unconstrained_waiting_time = np.sum(self.total_unconstrained_waiting_time[self.num_pad:self.num_resources - self.num_gate]) if self.num_resources - self.num_pad - self.num_gate > 0 else 0
        self.sum_total_unconstrained_waiting_time = np.sum(self.total_unconstrained_waiting_time) if self.num_resources > 0 else 0

        # Cost of delay index => Gate:Taxi:Airborne = 1:1.85:2.85
        self.cost_of_delay_index = self.gate_unconstrained_waiting_time + 1.85 * (self.pad_unconstrained_waiting_time + self.buffer_unconstrained_waiting_time) + 2.85 * self.total_arrival_tardiness

        # self.resource_utilization = self._compute_resource_utilization()
        
    def _get_operation_finish_time(self, vehicle: int, operation: int) -> float:
        """Get the actual finish time for an operation considering buffer operations"""
        if self.num_operations is None:
            return self.finish_times[vehicle, operation]
            
        if self.num_buffer_in is not None and self.num_buffer_out is not None:
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
        elif self.num_buffer is not None:
            if self.num_buffer > 0:
                if operation == 1:  # Buffer-In ends at Gate start
                    return self.start_times[vehicle, 2]
                elif operation == 3:  # Buffer-Out ends at Takeoff start
                    return self.start_times[vehicle, self.num_operations - 1]
        
        return self.finish_times[vehicle, operation]
    
    def get_operation_finish_time(self, vehicle: int, operation: int) -> float:
        """Public method to get the actual finish time for an operation considering buffer operations"""
        return self._get_operation_finish_time(vehicle, operation)
    
    def _get_operation_duration(self, vehicle: int, operation: int) -> float:
        """Get the duration of an operation"""
        start = self.start_times[vehicle, operation]
        finish = self._get_operation_finish_time(vehicle, operation)
        return max(0, finish - start)
    
    # def _compute_resource_utilization(self) -> Dict[int, float]:
    #     """Compute utilization percentage for each resource"""
    #     if not hasattr(self, 'num_resource') or self.num_vehicles is None or self.num_operations is None:
    #         return {}
    #
    #     utilization = {}
    #     total_time = np.max(self.finish_times) if self.finish_times.size > 0 else 1
    #
    #     for r in range(self.num_resources):
    #         busy_time = 0
    #         for v in range(self.num_vehicles):
    #             for o in range(self.num_operations):
    #                 if int(self.assigned_resources[v, o]) == r:
    #                     busy_time += self._get_operation_duration(v, o)
    #         utilization[r] = (busy_time / total_time) * 100 if total_time > 0 else 0
    #
    #     return utilization
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics for the solution"""
        stats = {
            'objective_value': self.obj_val,
            'runtime_seconds': self.solver_runtime,
            'total_arrival_tardiness': self.total_arrival_tardiness,
            'total_departure_tardiness': self.total_departure_tardiness,
            'total_tardiness': self.total_arrival_tardiness + self.total_departure_tardiness,
            'avg_arrival_tardiness': self.total_arrival_tardiness / self.num_vehicles if self.num_vehicles is not None and self.num_vehicles > 0 else 0,
            'avg_departure_tardiness': self.total_departure_tardiness / self.num_vehicles if self.num_vehicles is not None and self.num_vehicles > 0 else 0,
            'avg_total_tardiness': (self.total_arrival_tardiness + self.total_departure_tardiness) / self.num_vehicles if self.num_vehicles is not None and self.num_vehicles > 0 else 0,
            'max_arrival_tardiness': self.max_arrival_tardiness,
            'max_departure_tardiness': self.max_departure_tardiness,
            'sum_of_max_tardiness': self.max_arrival_tardiness + self.max_departure_tardiness,
            'max_vehicle_wise_tardiness': self.max_vehicle_wise_tardiness,
            'total_vehicles': self.num_vehicles if self.num_vehicles is not None else 0,
            'solver': self.solver_type,
            'simulation_end_time': self.sim_end_time,
        }
        
        # if hasattr(self, 'resource_utilization'):
        #     stats['avg_resource_utilization'] = np.mean(list(self.resource_utilization.values()))
        #     stats['max_resource_utilization'] = np.max(list(self.resource_utilization.values()))
            
        return stats

    def snapshot_at_time(
        self,
        t: float,
        tol: float = 1e-9,
        return_1_based_vehicle_ids: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a status snapshot at time `t`.

        Returns a dictionary with:
          - processing_vehicles: list of dicts (vehicles either actively processing or waiting),
                each item has:
                {
                  'vehicle': int,
                  'operation': int,          # op index
                  'resource': Optional[int], # resource index (if assigned)
                  'state': 'active' | 'waiting',
                  'start': Optional[float],  # op start (for active)
                  'finish': Optional[float], # op finish (for active)
                  'wait_type': Optional[str],# 'pre_first_operation' | 'between_operations' (for waiting)
                  'waiting_since': Optional[float],
                  'next_start': Optional[float]
                }
          - waiting_vehicles: subset of processing_vehicles with state == 'waiting'
          - counts: {
                'active': int,
                'waiting': int,
                'processing_total': int,
                'takeoff_finished_by_t': int,
                'takeoff_finishing_at_t_exact': int
            }
          - takeoff_finished_vehicles_by_t: List[int]
          - takeoff_finishing_at_t_exact: List[int]
        Notes
        -----
        • A vehicle is 'active' on operation o at t if start_times[v, o] <= t < adjusted_finish[v, o].
          The adjusted finish uses `get_operation_finish_time` so buffer operations extend until
          the next operation's start (consistent with your modeling).  # :contentReference[oaicite:1]{index=1}
        • 'waiting' means either:
            (a) the vehicle is ready (arrival <= t) but op 0 hasn't started by t, or
            (b) some op o finished by t, but op o+1 hasn't started by t.
          These are *reclassified* from processing-like into 'waiting'.
        • 'waiting_vehicles' is a strict subset of 'processing_vehicles'.
        """

        V = int(self.num_vehicles)
        O = int(self.num_operations)

        # Build buffer-aware finish matrix
        adjusted_finish = np.empty_like(self.finish_times, dtype=float)
        for v in range(V):
            for o in range(O):
                adjusted_finish[v, o] = self.get_operation_finish_time(v, o)  # buffer-aware  # :contentReference[oaicite:2]{index=2}

        start = self.start_times
        assigned = self.assigned_resources

        # --- 1) ACTUAL IN-SERVICE (active) ------------------------------------
        # We treat "active at t" as start <= t < finish (strictly before finish).
        in_service = (start <= (t + tol)) & (t < (adjusted_finish - tol))

        def _safe_int(x) -> Optional[int]:
            try:
                # Handle NaN or invalid resource entries gracefully
                if x is None:
                    return None
                if isinstance(x, (float, np.floating)) and np.isnan(x):
                    return None
                return int(x)
            except Exception:
                return None

        processing_vehicles: List[Dict[str, Any]] = []

        # Add active entries (may be 0, 1, or more per vehicle depending on data)
        active_vehicle_indices, active_op_indices = np.where(in_service)
        for v, o in zip(active_vehicle_indices, active_op_indices):
            res = _safe_int(assigned[v, o])
            item = {
                'vehicle': (v + 1) if return_1_based_vehicle_ids else v,
                'operation': int(o),
                'resource': res,
                'state': 'active',
                'start': float(start[v, o]),
                'finish': float(adjusted_finish[v, o]),
                'wait_type': None,
                'waiting_since': None,
                'next_start': None,
            }
            processing_vehicles.append(item)

        # --- 2) WAITING CLASSIFICATION ----------------------------------------
        # Waiting pre-first-op: ready (arrival) <= t but op 0 hasn't started by t
        arrival = getattr(self, 'ready', None)  # instance.vehicle_arrival_times  # :contentReference[oaicite:3]{index=3}
        waiting_pre_first = np.zeros(V, dtype=bool)
        if arrival is not None:
            arrival = np.asarray(arrival).reshape(-1)
            if arrival.shape[0] == V:
                waiting_pre_first = (arrival <= (t + tol)) & (start[:, 0] > (t + tol))

        # Waiting between ops: finish(o) <= t but start(o+1) > t
        waiting_next_op_index = np.full(V, -1, dtype=int)  # -1 => not waiting
        # Pre-first waiting takes precedence
        waiting_next_op_index[waiting_pre_first] = 0

        for o in range(O - 1):
            mask_o = (adjusted_finish[:, o] <= (t + tol)) & (start[:, o + 1] > (t + tol))
            # If we already assigned pre-first, keep it; otherwise assign the first gap found
            newly_waiting = mask_o & (waiting_next_op_index == -1)
            waiting_next_op_index[newly_waiting] = o + 1

        # Build waiting entries (subset of processing-like)
        for v in range(V):
            next_op = waiting_next_op_index[v]
            if next_op == -1:
                continue  # not waiting

            res_next = _safe_int(assigned[v, next_op])
            if next_op > 0:
                res_present = _safe_int(assigned[v, next_op - 1])
                res = res_present
            else:
                res = res_next
            pre_first = (next_op == 0) and waiting_pre_first[v]
            waiting_since = float(arrival[v]) if pre_first else float(adjusted_finish[v, next_op - 1])
            item = {
                'vehicle': (v + 1) if return_1_based_vehicle_ids else v,
                'operation': int(next_op - 1),             # the operation they are waiting to start
                'resource': res,                  # the resource assigned for that next op (if any)
                'state': 'waiting',
                'wait_type': 'pre_first_operation' if pre_first else 'between_operations',
                'waiting_since': waiting_since,
                'next_start': float(start[v, next_op]),
                'start': None,
                'finish': None,
            }
            processing_vehicles.append(item)

        # Extract strict waiting subset for convenience
        waiting_vehicles = [x for x in processing_vehicles if x['state'] == 'waiting']
        waiting_next_op_vehicles = [x for x in waiting_vehicles if x.get('wait_type') == 'between_operations']
        waiting_landing_vehicles = [x for x in waiting_vehicles if x.get('wait_type') == 'pre_first_operation']

        # Remove landing-waiting vehicles from processing list
        processing_vehicles = [
            p for p in processing_vehicles
            if p.get('wait_type') != 'pre_first_operation'
        ]

        # Extract operation and resource ids for processing vehicles
        processing_vehicles_id = [v['vehicle'] for v in processing_vehicles]
        processing_vehicles_operation = [v['operation'] for v in processing_vehicles]
        processing_vehicles_resource = [v['resource'] for v in processing_vehicles]

        # Extract operation and resource ids for waiting vehicles
        waiting_next_op_vehicles_id = [v['vehicle'] for v in waiting_next_op_vehicles]
        waiting_landing_vehicles_id = [v['vehicle'] for v in waiting_landing_vehicles]

        # --- 3) TAKEOFF COMPLETIONS -------------------------------------------
        last_op = O - 1
        last_finish = adjusted_finish[:, last_op]
        takeoff_finished_by_mask = last_finish <= (t + tol)
        takeoff_finished_at_exact_mask = np.isclose(last_finish, t, atol=tol)

        def _ids_from_mask(msk: np.ndarray) -> List[int]:
            vs = np.where(msk)[0].tolist()
            return [v + 1 for v in vs] if return_1_based_vehicle_ids else vs

        takeoff_finished_vehicles_by_t = _ids_from_mask(takeoff_finished_by_mask)
        takeoff_finishing_at_t_exact = _ids_from_mask(takeoff_finished_at_exact_mask)

        counts = {
            'active': sum(1 for x in processing_vehicles if x['state'] == 'active'),
            'waiting': len(waiting_vehicles),
            'waiting_next_op': len(waiting_next_op_vehicles),
            'waiting_landing': len(waiting_landing_vehicles),
            'processing_total': len(processing_vehicles),
            'takeoff_finished_by_t': int(np.sum(takeoff_finished_by_mask)),
            'takeoff_finishing_at_t_exact': int(np.sum(takeoff_finished_at_exact_mask)),
        }

        return {
            'processing_vehicles': processing_vehicles,
            'waiting_vehicles': waiting_vehicles,
            'waiting_next_op_vehicles': waiting_next_op_vehicles,
            'waiting_landing_vehicles': waiting_landing_vehicles,
            'processing_vehicles_ids': processing_vehicles_id,
            'processing_operation_ids': processing_vehicles_operation,
            'processing_resource_ids': processing_vehicles_resource,
            'waiting_next_op_vehicles_ids': waiting_next_op_vehicles_id,
            'waiting_landing_vehicles_ids': waiting_landing_vehicles_id,
            'counts': counts,
            'takeoff_finished_vehicles_by_t': takeoff_finished_vehicles_by_t,
            'takeoff_finishing_at_t_exact': takeoff_finishing_at_t_exact,
        }