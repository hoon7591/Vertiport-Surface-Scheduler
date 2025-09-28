import time
from abc import ABC, abstractmethod
from gurobipy import GRB, Model, quicksum
import numpy as np
import heapq
from typing import Dict, List, Tuple
from collections import defaultdict
from Solution import Solution
from Instance import Instance
from VertiportSimulator import VertiportSimulator, VertiportSimulatorRecedingHorizon,\
    EventType, Event, Vehicle, VehicleState, VehicleStateToOperation, Resource, ResourceState


class SolverStrategy(ABC):
    """Abstract base class for solver strategies."""
    
    @abstractmethod
    def solve(self, instance: Instance, is_numerical_exp: bool, **kwargs) -> Solution:
        """Solve the given instance and return a solution."""
        raise NotImplementedError("Subclasses must implement the solve method")
    
    def _setup_base_model(self, instance: Instance) -> Tuple[Model, Dict]:
        """Setup the base optimization model with common variables and constraints."""
        seed = instance.seed
        num_ops = instance.num_operations
        num_vehicle = instance.num_vehicles
        num_resource = instance.num_resource

        # Initialize Model
        model = Model("Vertiport_Surface_Scheduler")
        model.setParam('Seed', seed)
        model.setParam("OutputFlag", 1)

        # Define Variables
        S = model.addVars(num_ops, num_vehicle, vtype=GRB.CONTINUOUS, lb=0, name="start_time")
        y = model.addVars(num_ops, num_vehicle, num_resource, vtype=GRB.BINARY, name="y")
        x = model.addVars(num_ops, num_ops, num_vehicle, num_vehicle, num_resource, vtype=GRB.BINARY, name="x")
        T_a = model.addVars(num_vehicle, vtype=GRB.CONTINUOUS, lb=0, name="tardiness_arrival")
        T_d = model.addVars(num_vehicle, vtype=GRB.CONTINUOUS, lb=0, name="tardiness_departure")

        # Setup resource indices and add constraints
        resource_ind = self._setup_resource_indices(instance)
        self._add_base_constraints(model, instance, S, y, x, T_a, T_d, resource_ind)

        variables = {
            'S': S, 'y': y, 'x': x, 'T_a': T_a, 'T_d': T_d,
            'resource_ind': resource_ind
        }

        return model, variables

    def _setup_resource_indices(self, instance: Instance) -> List[List[int]]:
        """Setup resource indices based on instance configuration."""
        num_pad = instance.num_pad
        num_buffer_in = instance.num_buffer_in
        num_gate = instance.num_gate
        num_buffer_out = instance.num_buffer_out
        num_buffer = instance.num_buffer
        is_unified_buffer = instance.is_unified_buffer

        if is_unified_buffer:
            if num_buffer == 0:
                resource_ind = [[0, num_pad], [num_pad, num_pad + num_gate], [0, num_pad]]
            else:
                resource_ind = [[0, num_pad], [num_pad, num_pad + num_buffer],
                               [num_pad + num_buffer, num_pad + num_buffer + num_gate],
                               [num_pad, num_pad + num_buffer],
                               [0, num_pad]]
        else:
            if num_buffer_in == 0:
                if num_buffer_out == 0:
                    resource_ind = [[0, num_pad], [num_pad, num_pad + num_gate], [0, num_pad]]
                else:
                    resource_ind = [[0, num_pad], [num_pad, num_pad + num_gate],
                                    [num_pad + num_gate, num_pad + num_gate + num_buffer_out], [0, num_pad]]
            else:
                if num_buffer_out == 0:
                    resource_ind = [[0, num_pad], [num_pad, num_pad + num_buffer_in],
                                    [num_pad + num_buffer_in, num_pad + num_buffer_in + num_gate], [0, num_pad]]
                else:
                    resource_ind = [[0, num_pad], [num_pad, num_pad + num_buffer_in],
                                    [num_pad + num_buffer_in, num_pad + num_buffer_in + num_gate],
                                    [num_pad + num_buffer_in + num_gate, num_pad + num_buffer_in + num_gate + num_buffer_out],
                                    [0, num_pad]]
        return resource_ind

    def _add_base_constraints(self, model: Model, instance: Instance, S, y, x, T_a, T_d, resource_ind: List[List[int]]):
        """Add base constraints common to all solver strategies."""
        num_ops = instance.num_operations
        num_vehicle = instance.num_vehicles
        num_buffer_out = instance.num_buffer_out
        num_buffer = instance.num_buffer
        is_unified_buffer = instance.is_unified_buffer
        vehicle_arrival_times = instance.vehicle_arrival_times
        processing_times = instance.proc
        vehicle_planned_arrival_times = instance.vehicle_planned_arrival_times
        vehicle_planned_departure_times = instance.vehicle_planned_departure_times
        vehicle_planned_gate_close_times = instance.vehicle_planned_gate_close_times
        ST = instance.ST
        big_M = instance.big_M

        # Const. 1: Vehicle Assignment
        for i in range(num_ops):
            for j in range(num_vehicle):
                model.addConstr(quicksum(y[i, j, k] for k in range(resource_ind[i][0], resource_ind[i][1])) == 1)

        # Const. 2: Precedence
        for i in range(num_ops - 1):
            for j in range(num_vehicle):
                model.addConstr(S[i + 1, j] >= S[i, j] + quicksum(y[i, j, k] * processing_times[i][j][k - resource_ind[i][0]]
                                                                  for k in range(resource_ind[i][0], resource_ind[i][1])))

        # Const. 3: Overlap Avoidance with Reentrant Condition and Blocking
        for i in range(num_ops):
            for i_ in range(num_ops):
                res_i = list(range(resource_ind[i][0], resource_ind[i][1]))
                res_i_ = list(range(resource_ind[i_][0], resource_ind[i_][1]))
                res_intersect = list(set(res_i) & set(res_i_))
                if len(res_intersect) == 0:
                    continue
                for j in range(num_vehicle):
                    for j_ in range(num_vehicle):
                        if j != j_:
                            for k in res_intersect:
                                model.addConstr(x[i, i_, j, j_, k] <= y[i, j, k])
                                model.addConstr(x[i, i_, j, j_, k] <= y[i_, j_, k])
                                model.addConstr(x[i, i_, j, j_, k] + x[i_, i, j_, j, k] <= 1)
                                model.addConstr(x[i, i_, j, j_, k] + x[i_, i, j_, j, k] >= y[i, j, k] + y[i_, j_, k] - 1)
                                if i != num_ops - 1:
                                    model.addConstr(S[i + 1, j] <= S[i_, j_] + big_M * (1 - x[i, i_, j, j_, k]))
                                else:
                                    model.addConstr(S[i, j] + processing_times[i][j][k - resource_ind[i][0]] <= S[i_, j_] + big_M * (1 - x[i, i_, j, j_, k]))

                                # Const. 4: Separation
                                if (i == 0 or i == num_ops - 1) and (i_ == 0 or i_ == num_ops - 1):
                                    model.addConstr(S[i_, j_] >= S[i, j] + processing_times[i][j][k - resource_ind[i][0]] +
                                                    ST[(i, i_)][j][j_][k] - big_M * (1 - x[i, i_, j, j_, k]))

        # Const. 5: Ready
        for i in range(num_vehicle):
            model.addConstr(S[0, i] >= vehicle_arrival_times[i])

        # Const. 6: Tardiness Calculation with No early Departure
        for i in range(num_vehicle):
            model.addConstr(T_a[i] >= S[0, i] + quicksum(y[0, i, j] * processing_times[0][i][j - resource_ind[0][0]]
                                                         for j in range(resource_ind[0][0], resource_ind[0][1])) - vehicle_planned_arrival_times[i])
            if is_unified_buffer:
                if num_buffer > 0:
                    model.addConstr(S[num_ops - 2, i] >= vehicle_planned_gate_close_times[i])
                else:
                    model.addConstr(S[num_ops - 1, i] >= vehicle_planned_gate_close_times[i])
            else:
                if num_buffer_out > 0:
                    model.addConstr(S[num_ops - 2, i] >= vehicle_planned_gate_close_times[i])
                else:
                    model.addConstr(S[num_ops - 1, i] >= vehicle_planned_gate_close_times[i])

            model.addConstr(T_d[i] >= S[num_ops - 1, i] - vehicle_planned_departure_times[i])

    def _extract_solution(self, model: Model, instance: Instance, variables: Dict, solver_name: str, is_deadlock, is_runtime_over) -> Solution:
        """Extract solution from the solved model."""
        num_ops = instance.num_operations
        num_vehicle = instance.num_vehicles
        weights = instance.objective_weights
        
        S = variables['S']
        y = variables['y']
        T_a = variables['T_a']
        T_d = variables['T_d']
        resource_ind = variables['resource_ind']
        proc = instance.proc

        # Extract solution arrays
        start_times = np.zeros((num_vehicle, num_ops))
        finish_times = np.zeros((num_vehicle, num_ops))
        assigned_resources = np.zeros((num_vehicle, num_ops))
        arrival_time_tardiness = np.zeros(num_vehicle)
        departure_time_tardiness = np.zeros(num_vehicle)

        obj_val = model.ObjVal if hasattr(model, 'ObjVal') else 0
        runtime = model.Runtime if hasattr(model, 'Runtime') else 0

        for i in range(num_vehicle):
            arrival_time_tardiness[i] = T_a[i].X
            departure_time_tardiness[i] = T_d[i].X
            for j in range(num_ops):
                start_times[i, j] = S[j, i].X
                finish_times[i, j] = start_times[i, j]
                for k in range(resource_ind[j][0], resource_ind[j][1]):
                    if y[j, i, k].X >= 0.5:
                        finish_times[i, j] += proc[j][i][k - resource_ind[j][0]]
                        assigned_resources[i, j] = k

        sim_end_time = np.max(finish_times)

        # Calculate objective value for SAT-based solvers
        if solver_name in ["FCFS_SAT", "FCFS_landing_SAT", "no_rule_SAT"]:
            obj_val = weights[0] * sum(arrival_time_tardiness) + weights[1] * sum(departure_time_tardiness)

        return Solution(obj_val, runtime, sim_end_time, start_times, finish_times, assigned_resources,
                        arrival_time_tardiness, departure_time_tardiness, resource_ind, solver_name, instance,
                        is_deadlock, is_runtime_over)


class SolverStrategyRHC(SolverStrategy):
    @abstractmethod
    def solve(self, instance: Instance, is_numerical_exp: bool, processing_vehicles_op: List[int],
              processing_vehicles_res: List[int], horizon_start: float) -> Solution:
        """Solve the given instance in receding horizon control context and return a solution."""

        raise NotImplementedError("Subclasses must implement the solve method")

    def _setup_base_model(self, instance: Instance, processing_vehicles_op: List[int], processing_vehicles_res: List[int]) -> Tuple[Model, Dict]:
        """Setup the base optimization model with common variables and constraints."""
        seed = instance.seed
        num_ops = instance.num_operations
        num_vehicle = instance.num_vehicles
        num_resource = instance.num_resource

        # Initialize Model
        model = Model("Vertiport_Surface_Scheduler")
        model.setParam('Seed', seed)
        model.setParam("OutputFlag", 1)

        # Define Variables
        S = model.addVars(num_ops, num_vehicle, vtype=GRB.CONTINUOUS, lb=0, name="start_time")
        y = model.addVars(num_ops, num_vehicle, num_resource, vtype=GRB.BINARY, name="y")
        x = model.addVars(num_ops, num_ops, num_vehicle, num_vehicle, num_resource, vtype=GRB.BINARY, name="x")
        T_a = model.addVars(num_vehicle, vtype=GRB.CONTINUOUS, lb=0, name="tardiness_arrival")
        T_d = model.addVars(num_vehicle, vtype=GRB.CONTINUOUS, lb=0, name="tardiness_departure")

        # Setup resource indices and add constraints
        resource_ind = self._setup_resource_indices(instance)
        self._add_base_constraints(model, instance, S, y, x, T_a, T_d, resource_ind, processing_vehicles_op, processing_vehicles_res)

        variables = {
            'S': S, 'y': y, 'x': x, 'T_a': T_a, 'T_d': T_d,
            'resource_ind': resource_ind
        }

        return model, variables

    def _add_base_constraints(self, model: Model, instance: Instance, S, y, x, T_a, T_d, resource_ind: List[List[int]],
                              processing_vehicles_op: List[int], processing_vehicles_res: List[int]):
        """Add base constraints common to all solver strategies."""
        num_ops = instance.num_operations
        num_vehicle = instance.num_vehicles
        num_buffer_out = instance.num_buffer_out
        num_buffer = instance.num_buffer
        is_unified_buffer = instance.is_unified_buffer
        vehicle_arrival_times = instance.vehicle_arrival_times
        processing_times = instance.proc
        vehicle_planned_arrival_times = instance.vehicle_planned_arrival_times
        vehicle_planned_departure_times = instance.vehicle_planned_departure_times
        vehicle_planned_gate_close_times = instance.vehicle_planned_gate_close_times
        ST = instance.ST
        big_M = instance.big_M

        # Const. 1: Vehicle Assignment
        for i in range(num_ops):
            for j in range(num_vehicle):
                model.addConstr(quicksum(y[i, j, k] for k in range(resource_ind[i][0], resource_ind[i][1])) == 1)

        # Const. 2: Precedence
        for i in range(num_ops - 1):
            for j in range(num_vehicle):
                model.addConstr(
                    S[i + 1, j] >= S[i, j] + quicksum(y[i, j, k] * processing_times[i][j][k - resource_ind[i][0]]
                                                      for k in range(resource_ind[i][0], resource_ind[i][1])))

        # Const. 3: Overlap Avoidance with Reentrant Condition and Blocking
        for i in range(num_ops):
            for i_ in range(num_ops):
                res_i = list(range(resource_ind[i][0], resource_ind[i][1]))
                res_i_ = list(range(resource_ind[i_][0], resource_ind[i_][1]))
                res_intersect = list(set(res_i) & set(res_i_))
                if len(res_intersect) == 0:
                    continue
                for j in range(num_vehicle):
                    for j_ in range(num_vehicle):
                        if j != j_:
                            if j < len(processing_vehicles_op) and j_ < len(processing_vehicles_op):
                                if i < processing_vehicles_op[j] and i_ < processing_vehicles_op[j_]:
                                    continue
                                else:
                                    for k in res_intersect:
                                        self._add_overlap_separation_constraints(model, S, y, x, resource_ind,
                                                                                 i, i_, j, j_, k, num_ops, processing_times,
                                                                                 ST, big_M)
                            else:
                                for k in res_intersect:
                                    self._add_overlap_separation_constraints(model, S, y, x, resource_ind,
                                                                             i, i_, j, j_, k, num_ops, processing_times,
                                                                             ST, big_M)

        # Const. 5: Ready
        for i in range(num_vehicle):
            model.addConstr(S[0, i] >= vehicle_arrival_times[i])

        # Const. 6: Tardiness Calculation with No early Departure
        for i in range(num_vehicle):
            model.addConstr(T_a[i] >= S[0, i] + quicksum(y[0, i, j] * processing_times[0][i][j - resource_ind[0][0]]
                                                         for j in range(resource_ind[0][0], resource_ind[0][1])) -
                            vehicle_planned_arrival_times[i])
            if is_unified_buffer:
                if num_buffer > 0:
                    model.addConstr(S[num_ops - 2, i] >= vehicle_planned_gate_close_times[i])
                else:
                    model.addConstr(S[num_ops - 1, i] >= vehicle_planned_gate_close_times[i])
            else:
                if num_buffer_out > 0:
                    model.addConstr(S[num_ops - 2, i] >= vehicle_planned_gate_close_times[i])
                else:
                    model.addConstr(S[num_ops - 1, i] >= vehicle_planned_gate_close_times[i])

            model.addConstr(T_d[i] >= S[num_ops - 1, i] - vehicle_planned_departure_times[i])

    def _add_overlap_separation_constraints(self, model: Model, S, y, x, resource_ind: List[List[int]],
                                            i, i_, j, j_, k, num_ops, processing_times, ST, big_M):
        model.addConstr(x[i, i_, j, j_, k] <= y[i, j, k])
        model.addConstr(x[i, i_, j, j_, k] <= y[i_, j_, k])
        model.addConstr(x[i, i_, j, j_, k] + x[i_, i, j_, j, k] <= 1)
        model.addConstr(
        x[i, i_, j, j_, k] + x[i_, i, j_, j, k] >= y[i, j, k] + y[i_, j_, k] - 1)
        if i != num_ops - 1:
            model.addConstr(S[i + 1, j] <= S[i_, j_] + big_M * (1 - x[i, i_, j, j_, k]))
        else:
            model.addConstr(
        S[i, j] + processing_times[i][j][k - resource_ind[i][0]] <= S[
            i_, j_] + big_M * (1 - x[i, i_, j, j_, k]))

        # Const. 4: Separation
        if (i == 0 or i == num_ops - 1) and (i_ == 0 or i_ == num_ops - 1):
            model.addConstr(
        S[i_, j_] >= S[i, j] + processing_times[i][j][k - resource_ind[i][0]] +
        ST[(i, i_)][j][j_][k] - big_M * (1 - x[i, i_, j, j_, k]))


class RunStrategyRHC(SolverStrategy):
    @abstractmethod
    def solve(self, instance: Instance, is_numerical_exp: bool, planned_resource_assignment: np.ndarray,
              planned_operation_start_times: np.ndarray, vehicle_original_id: List[int]) -> Solution:
        raise NotImplementedError("Subclasses must implement the solve method")


class ExactSolver(SolverStrategy):
    """Exact solver that guarantees optimal solution."""

    def solve(self, instance: Instance, is_numerical_exp: bool) -> Solution:
        model, variables = self._setup_base_model(instance)

        # Set objective for exact solver
        weights = instance.objective_weights
        num_vehicle = instance.num_vehicles
        T_a = variables['T_a']
        T_d = variables['T_d']

        model.setObjective(quicksum(weights[0] * T_a[i] + weights[1] * T_d[i] for i in range(num_vehicle)), GRB.MINIMIZE)

        # Solve the model
        if is_numerical_exp:
            model.setParam('TimeLimit', 100)
            model.optimize()
            if model.Status == GRB.TIME_LIMIT:
                is_runtime_over = True
            elif model.Status == GRB.OPTIMAL:
                is_runtime_over = False
        else:
            model.optimize()
            is_runtime_over = False

        is_deadlock = False

        return self._extract_solution(model, instance, variables, "exact", is_deadlock, is_runtime_over)


class ExactSolverRHC(SolverStrategyRHC):
    def solve(self, instance: Instance, is_numerical_exp: bool, processing_vehicles_op: List[int],
              processing_vehicles_res: List[int], horizon_start: float) -> Solution:
        model, variables = self._setup_base_model(instance, processing_vehicles_op, processing_vehicles_res)

        # Set objective for exact solver
        weights = instance.objective_weights
        num_vehicle = instance.num_vehicles
        T_a = variables['T_a']
        T_d = variables['T_d']

        model.setObjective(quicksum(weights[0] * T_a[i] + weights[1] * T_d[i] for i in range(num_vehicle)), GRB.MINIMIZE)

        # Add constraints for processing vehicles
        y = variables['y']
        S = variables['S']

        for i in range(len(processing_vehicles_op)):
            model.addConstr(y[processing_vehicles_op[i], i, processing_vehicles_res[i]] == 1)
            model.addConstr(S[processing_vehicles_op[i], i] == horizon_start)

        # Solve the model
        if is_numerical_exp:
            model.setParam('TimeLimit', 100)
            model.optimize()
            if model.Status == GRB.TIME_LIMIT:
                is_runtime_over = True
            elif model.Status == GRB.OPTIMAL:
                is_runtime_over = False
        else:
            model.optimize()
            is_runtime_over = False

        is_deadlock = False

        return self._extract_solution(model, instance, variables, "exact_RHC", is_deadlock, is_runtime_over)


class FCFSSolver(SolverStrategy):
    """First-Come-First-Served solver implementation."""

    def __init__(self, is_objective_enabled: bool = False, FCFS_for_landing_only: bool = False):
        self.use_gurobi = is_objective_enabled
        self.landing_only = FCFS_for_landing_only

    def solve(self, instance: Instance, is_numerical_exp: bool) -> Solution:
        model, variables = self._setup_base_model(instance)

        # Set objective only for Gurobi variant
        if self.use_gurobi:
            weights = instance.objective_weights
            num_vehicle = instance.num_vehicles
            T_a = variables['T_a']
            T_d = variables['T_d']
            model.setObjective(quicksum(weights[0] * T_a[i] + weights[1] * T_d[i] for i in range(num_vehicle)), GRB.MINIMIZE)

        # Add FCFS constraints
        if self.landing_only:
            self._add_fcfs_landing_constraints(model, instance, variables)
            solver_name = "FCFS_landing_Gurobi" if self.use_gurobi else "FCFS_landing_SAT"
        else:
            self._add_fcfs_constraints(model, instance, variables)
            solver_name = "FCFS_Gurobi" if self.use_gurobi else "FCFS_SAT"

        # Solve the model
        if is_numerical_exp:
            model.setParam('TimeLimit', 100)
            model.optimize()
            if model.Status == GRB.TIME_LIMIT:
                is_runtime_over = True
            elif model.Status == GRB.OPTIMAL:
                is_runtime_over = False
        else:
            model.optimize()
            is_runtime_over = False

        is_deadlock = False

        return self._extract_solution(model, instance, variables, solver_name, is_deadlock, is_runtime_over)

    def _add_fcfs_constraints(self, model, instance: Instance, variables):
        """Add FCFS-specific constraints for all operations."""
        num_ops = instance.num_operations
        num_vehicle = instance.num_vehicles
        ready = instance.vehicle_arrival_times
        S = variables['S']
        x = variables['x']
        resource_ind = variables['resource_ind']

        sorted_vehicle_indices = np.argsort(ready)
        for i in range(num_ops):
            for j in range(num_vehicle - 1):
                model.addConstr(S[i, sorted_vehicle_indices[j]] <= S[i, sorted_vehicle_indices[j + 1]])
        for i in range(num_ops):
            for j in range(num_vehicle - 1):
                for j_ in range(j + 1, num_vehicle):
                    for k in range(resource_ind[i][0], resource_ind[i][1]):
                        model.addConstr(x[i, i, sorted_vehicle_indices[j_], sorted_vehicle_indices[j], k] == 0)

    def _add_fcfs_landing_constraints(self, model, instance: Instance, variables):
        """Add FCFS landing-specific constraints."""
        num_vehicle = instance.num_vehicles
        ready = instance.vehicle_arrival_times
        S = variables['S']
        x = variables['x']
        resource_ind = variables['resource_ind']

        sorted_vehicle_indices = np.argsort(ready)
        for j in range(num_vehicle - 1):
            model.addConstr(S[0, sorted_vehicle_indices[j]] <= S[0, sorted_vehicle_indices[j + 1]])
        for j in range(num_vehicle - 1):
            for j_ in range(j + 1, num_vehicle):
                for k in range(resource_ind[0][0], resource_ind[0][1]):
                    model.addConstr(x[0, 0, sorted_vehicle_indices[j_], sorted_vehicle_indices[j], k] == 0)


class NoRuleSolver(SolverStrategy):
    """SAT solver without any primary sequencing rule."""

    def solve(self, instance: Instance, is_numerical_exp: bool) -> Solution:
        model, variables = self._setup_base_model(instance)

        # No objective function set for SAT mode
        if is_numerical_exp:
            model.setParam('TimeLimit', 100)
            model.optimize()
            if model.Status == GRB.TIME_LIMIT:
                is_runtime_over = True
            elif model.Status == GRB.OPTIMAL:
                is_runtime_over = False
        else:
            model.optimize()
            is_runtime_over = False

        is_deadlock = False

        return self._extract_solution(model, instance, variables, "no_rule_SAT", is_deadlock, is_runtime_over)


class FCFS_HeuristicSolver(SolverStrategy):
    def solve(self, instance: Instance, is_numerical_exp: bool) -> Solution:
        # Create simulator
        simulator = VertiportSimulator(instance, is_numerical_exp)
        is_deadlock = False
        is_runtime_over = False

        # Step-by-step control with FCFS logic
        solve_start_time = time.time()
        while not simulator.is_simulation_complete():
            event = simulator.step_to_next_event()

            # FCFS scheduling logic here - TODO : improve more - multiple vehicles and multiple resources
            for operation in range(instance.num_operations):
                waiting_vehicles = simulator.get_waiting_vehicles(operation)
                available_resources = simulator.get_available_resources(operation)

                # Make assignment decisions
                if waiting_vehicles and available_resources:
                    break_flag = False
                    for i in range(len(waiting_vehicles)):
                        for j in range(len(available_resources)):
                            vehicle = waiting_vehicles[i]  # selection logic
                            resource = available_resources[j]  # selection logic

                            if simulator.can_assign_vehicle_to_resource(vehicle.id, resource.id, operation):
                                simulator.assign_vehicle_to_resource_now(vehicle.id, resource.id, operation)
                                break_flag = True
                                break
                            # Error fixing of separation time reflection; only the worst case separation is considered before
                            elif (event.event_type == EventType.RESOURCE_AVAILABLE or event.event_type == EventType.VEHICLE_ARRIVAL or event.event_type == EventType.OPERATION_COMPLETE) and resource.state == ResourceState.SEPARATION_DELAY:
                                separation_clear_event = Event(time=simulator.current_time,
                                                               event_type=EventType.RESOURCE_AVAILABLE,
                                                               vehicle_id=vehicle.id,
                                                               resource_id=resource.id,
                                                               operation_id=operation,
                                                               event_id=event.event_id)
                                heapq.heappush(simulator.event_queue, separation_clear_event)
                                resource.state = ResourceState.IDLE

            if event is None and simulator.is_simulation_complete() is False and is_numerical_exp:
                is_deadlock = True
                break

        solve_end_time = time.time()
        runtime = solve_end_time - solve_start_time

        return simulator._generate_solution(runtime, is_deadlock, is_runtime_over, solver_type="FCFS_heuristic")


class RunRHC(RunStrategyRHC):
    def solve(self, instance: Instance, is_numerical_exp: bool, planned_resource_assignment: np.ndarray,
              planned_operation_start_times: np.ndarray, vehicle_original_id: List[int]) -> Solution:

        # Create simulator
        simulator = VertiportSimulatorRecedingHorizon(instance, is_numerical_exp)
        is_deadlock = False
        is_runtime_over = False

        # Step-by-step control with RunRHC logic
        solve_start_time = time.time()
        while not simulator.is_simulation_complete():
            event = simulator.step_to_next_event()

            # RunRHC logic here - TODO : improve more - multiple vehicles and multiple resources
            for operation in range(instance.num_operations):
                waiting_vehicles = simulator.get_waiting_vehicles(operation)
                available_resources = simulator.get_available_resources(operation)

                # Make assignment decisions
                if waiting_vehicles and available_resources:
                    break_flag = False
                    for i in range(len(waiting_vehicles)):
                        for j in range(len(available_resources)):
                            vehicle = waiting_vehicles[i]  # selection logic
                            resource = available_resources[j]  # selection logic
                            if resource.id == planned_resource_assignment[vehicle_original_id[vehicle.id], operation]:
                                if simulator.current_time >= planned_operation_start_times[vehicle_original_id[vehicle.id], operation]:
                                    if simulator.can_assign_vehicle_to_resource(vehicle.id, resource.id, operation):
                                        simulator.assign_vehicle_to_resource_now(vehicle.id, resource.id, operation)
                                        break_flag = True
                                        break
                                    elif (event.event_type == EventType.RESOURCE_AVAILABLE or event.event_type == EventType.VEHICLE_ARRIVAL or event.event_type == EventType.OPERATION_COMPLETE) and resource.state == ResourceState.SEPARATION_DELAY:
                                        separation_clear_event = Event(time=simulator.current_time,
                                                                       event_type=EventType.RESOURCE_AVAILABLE,
                                                                       vehicle_id=vehicle.id,
                                                                       resource_id=resource.id,
                                                                       operation_id=operation,
                                                                       event_id=event.event_id)
                                        heapq.heappush(simulator.event_queue, separation_clear_event)
                                        resource.state = ResourceState.IDLE
                                else:
                                    if operation == 0 and event.event_type == EventType.VEHICLE_ARRIVAL:
                                        arrival_event = Event(time=planned_operation_start_times[vehicle_original_id[vehicle.id], operation],
                                                              event_type=EventType.RESOURCE_AVAILABLE,
                                                              vehicle_id=vehicle.id,
                                                              resource_id=resource.id,
                                                              operation_id=operation,
                                                              event_id=event.event_id)
                                        heapq.heappush(simulator.event_queue, arrival_event)
                                        vehicle.state = VehicleState.WAITING_FOR_LANDING
                                    elif event.event_type == EventType.OPERATION_COMPLETE:
                                        completion_event = Event(time=planned_operation_start_times[vehicle_original_id[vehicle.id], operation],
                                                                 event_type=EventType.RESOURCE_AVAILABLE,
                                                                 vehicle_id=vehicle.id,
                                                                 resource_id=resource.id,
                                                                 operation_id=operation,
                                                                 event_id=event.event_id)
                                        heapq.heappush(simulator.event_queue, completion_event)
                                        inverse_mapping = defaultdict(list)
                                        for k, v in VehicleStateToOperation.mapping.items():
                                            inverse_mapping[v].append(k)
                                        vehicle.state = inverse_mapping[operation][0]
                        if break_flag:
                            break

            if event is None and simulator.is_simulation_complete() is False and is_numerical_exp:
                is_deadlock = True
                break

        solve_end_time = time.time()
        runtime = solve_end_time - solve_start_time

        return simulator._generate_solution(runtime, is_deadlock, is_runtime_over, solver_type="run_RHC")


# Solver factory with minimal overhead
_SOLVER_INSTANCES = {
    "exact": ExactSolver(),
    "FCFS_SAT": FCFSSolver(is_objective_enabled=False, FCFS_for_landing_only=False),
    "FCFS_Gurobi": FCFSSolver(is_objective_enabled=True, FCFS_for_landing_only=False),
    "FCFS_landing_SAT": FCFSSolver(is_objective_enabled=False, FCFS_for_landing_only=True),
    "FCFS_landing_Gurobi": FCFSSolver(is_objective_enabled=True, FCFS_for_landing_only=True),
    "no_rule_SAT": NoRuleSolver(),
    "FCFS_heuristic": FCFS_HeuristicSolver(),
    "exact_RHC": ExactSolverRHC(),
    "run_RHC": RunRHC(),
}


def solve(instance: Instance, solver: str, is_numerical_exp: bool, **kwargs) -> Solution:
    """
    Solve the vertiport surface scheduling problem using the specified solver strategy.

    This function serves as the main entry point for solving optimization problems.
    It uses the Strategy pattern with pre-instantiated solvers for optimal performance.

    Args:
        instance: Instance object containing problem data
        solver: String identifier for the solver strategy to use
                Available options: "exact", "FCFS_Gurobi", "FCFS_landing_Gurobi",
                "FCFS_SAT", "FCFS_landing_SAT", "no_rule_SAT", "FCFS_heuristic"
    
    Returns:
        Solution: Solution object containing optimization results
        
    Raises:
        ValueError: If solver type is not recognized
    """
    if solver not in _SOLVER_INSTANCES:
        available_solvers = ", ".join(_SOLVER_INSTANCES.keys())
        raise ValueError(f"Unknown solver type: '{solver}'. Available solvers: {available_solvers}")
    
    return _SOLVER_INSTANCES[solver].solve(instance, is_numerical_exp, **kwargs)
