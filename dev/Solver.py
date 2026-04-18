import time
from abc import ABC, abstractmethod
from gurobipy import GRB, Model, quicksum
import numpy as np
import heapq
from typing import Dict, List, Tuple, Optional
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

    def _setup_base_model(
            self,
            instance: Instance,
            processing_vehicles_op: Optional[List[int]] = None,
            processing_vehicles_res: Optional[List[int]] = None,
    ) -> Tuple[Model, Dict]:
        """
        Setup the base optimization model with common variables and constraints.

        processing_vehicles_op / processing_vehicles_res 가 주어지면
        (RHC 맥락) 거기에 맞게 overlap 제약을 일부 스킵하는 모드로 동작하고,
        None 이면 기존 exact / FCFS / no_rule 과 동일하게 동작합니다.
        """
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
        resource_ind, landing_op_id, buffer_in_op_id, gate_op_id, buffer_out_op_id, takeoff_op_id = self._setup_resource_indices(instance)
        self._add_base_constraints(
            model,
            instance,
            S, y, x, T_a, T_d,
            resource_ind,
            processing_vehicles_op=processing_vehicles_op,
            processing_vehicles_res=processing_vehicles_res,
        )

        variables = {
            'S': S, 'y': y, 'x': x, 'T_a': T_a, 'T_d': T_d,
            'resource_ind': resource_ind
        }

        return model, variables

    def _setup_resource_indices(self, instance: Instance) -> Tuple[List[List[int]], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Setup resource indices based on instance configuration."""
        num_pad = instance.num_pad
        num_buffer_in = instance.num_buffer_in
        num_gate = instance.num_gate
        num_buffer_out = instance.num_buffer_out
        num_buffer = instance.num_buffer
        is_unified_buffer = instance.is_unified_buffer

        landing_op_id = None
        buffer_in_op_id = None
        gate_op_id = None
        burfer_out_op_id = None
        takeoff_op_id = None

        if is_unified_buffer:
            if num_buffer == 0:
                resource_ind = [[0, num_pad], [num_pad, num_pad + num_gate], [0, num_pad]]
                landing_op_id = 0
                gate_op_id = 1
                takeoff_op_id = 2
            else:
                resource_ind = [
                    [0, num_pad],
                    [num_pad, num_pad + num_buffer],
                    [num_pad + num_buffer, num_pad + num_buffer + num_gate],
                    [num_pad, num_pad + num_buffer],
                    [0, num_pad],
                ]
                landing_op_id = 0
                buffer_in_op_id = 1
                gate_op_id = 2
                buffer_out_op_id = 3
                takeoff_op_id = 4
        else:
            if num_buffer_in == 0:
                if num_buffer_out == 0:
                    resource_ind = [[0, num_pad], [num_pad, num_pad + num_gate], [0, num_pad]]
                    landing_op_id = 0
                    gate_op_id = 1
                    takeoff_op_id = 2
                else:
                    resource_ind = [
                        [0, num_pad],
                        [num_pad, num_pad + num_gate],
                        [num_pad + num_gate, num_pad + num_gate + num_buffer_out],
                        [0, num_pad],
                    ]
                    landing_op_id = 0
                    gate_op_id = 1
                    buffer_out_op_id = 2
                    takeoff_op_id = 3
            else:
                if num_buffer_out == 0:
                    resource_ind = [
                        [0, num_pad],
                        [num_pad, num_pad + num_buffer_in],
                        [num_pad + num_buffer_in, num_pad + num_buffer_in + num_gate],
                        [0, num_pad],
                    ]
                    landing_op_id = 0
                    buffer_in_op_id = 1
                    gate_op_id = 2
                    takeoff_op_id = 3
                else:
                    resource_ind = [
                        [0, num_pad],
                        [num_pad, num_pad + num_buffer_in],
                        [num_pad + num_buffer_in, num_pad + num_buffer_in + num_gate],
                        [num_pad + num_buffer_in + num_gate,
                         num_pad + num_buffer_in + num_gate + num_buffer_out],
                        [0, num_pad],
                    ]
                    landing_op_id = 0
                    buffer_in_op_id = 1
                    gate_op_id = 2
                    buffer_out_op_id = 3
                    takeoff_op_id = 4

        return resource_ind, landing_op_id, buffer_in_op_id, gate_op_id, buffer_out_op_id, takeoff_op_id

    def _add_base_constraints(
            self,
            model: Model,
            instance: Instance,
            S, y, x, T_a, T_d,
            resource_ind: List[List[int]],
            processing_vehicles_op: Optional[List[int]] = None,
            processing_vehicles_res: Optional[List[int]] = None,
    ):
        """
        기본 MILP 제약들을 추가.

        processing_vehicles_op 이 None 이면 기존 SolverStrategy 동작과 동일하고,
        리스트가 주어지면 이전의 SolverStrategyRHC 와 같은 방식으로
        overlap 제약을 일부 스킵하도록 동작합니다.
        """
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
                model.addConstr(
                    quicksum(y[i, j, k] for k in range(resource_ind[i][0], resource_ind[i][1])) == 1
                )

        # Const. 2: Precedence
        for i in range(num_ops - 1):
            for j in range(num_vehicle):
                model.addConstr(
                    S[i + 1, j]
                    >= S[i, j]
                    + quicksum(
                        y[i, j, k] * processing_times[i][j][k - resource_ind[i][0]]
                        for k in range(resource_ind[i][0], resource_ind[i][1])
                    )
                )

        # Const. 3: Overlap Avoidance with Reentrant Condition and Blocking
        for i in range(num_ops):
            for i_ in range(num_ops):
                res_i = list(range(resource_ind[i][0], resource_ind[i][1]))
                res_i_ = list(range(resource_ind[i_][0], resource_ind[i_][1]))
                res_intersect = list(set(res_i) & set(res_i_))
                if not res_intersect:
                    continue

                for j in range(num_vehicle):
                    for j_ in range(num_vehicle):
                        if j == j_:
                            continue

                        # RHC 모드일 때: 이미 과거에 처리된 operation 쌍에 대해서는 overlap 제약을 스킵
                        if processing_vehicles_op is not None:
                            if (
                                    j < len(processing_vehicles_op)
                                    and j_ < len(processing_vehicles_op)
                                    and i < processing_vehicles_op[j]
                                    and i_ < processing_vehicles_op[j_]
                            ):
                                # 이 경우는 이전 SolverStrategyRHC에서 continue 하던 케이스
                                continue

                        for k in res_intersect:
                            self._add_overlap_separation_constraints(
                                model, S, y, x, resource_ind,
                                i, i_, j, j_, k,
                                num_ops, processing_times, ST, big_M,
                            )

        # Const. 5: Ready
        for i in range(num_vehicle):
            model.addConstr(S[0, i] >= vehicle_arrival_times[i])

        # Const. 6: Tardiness Calculation with No early Departure
        for i in range(num_vehicle):
            model.addConstr(
                T_a[i]
                >= S[0, i]
                + quicksum(
                    y[0, i, j] * processing_times[0][i][j - resource_ind[0][0]]
                    for j in range(resource_ind[0][0], resource_ind[0][1])
                )
                - vehicle_planned_arrival_times[i]
            )

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

            model.addConstr(
                T_d[i] >= S[num_ops - 1, i] - vehicle_planned_departure_times[i]
            )

    def _add_overlap_separation_constraints(
            self,
            model: Model,
            S, y, x,
            resource_ind: List[List[int]],
            i, i_, j, j_, k,
            num_ops, processing_times, ST, big_M,
    ):
        """
        기존 SolverStrategy._add_base_constraints 와 SolverStrategyRHC._add_overlap_separation_constraints
        에서 공통으로 쓰이던 overlap + separation 제약을 하나의 helper 로 모은 것.
        """
        model.addConstr(x[i, i_, j, j_, k] <= y[i, j, k])
        model.addConstr(x[i, i_, j, j_, k] <= y[i_, j_, k])
        model.addConstr(x[i, i_, j, j_, k] + x[i_, i, j_, j, k] <= 1)
        model.addConstr(
            x[i, i_, j, j_, k] + x[i_, i, j_, j, k] >= y[i, j, k] + y[i_, j_, k] - 1
        )

        if i != num_ops - 1:
            model.addConstr(
                S[i + 1, j]
                <= S[i_, j_] + big_M * (1 - x[i, i_, j, j_, k])
            )
        else:
            model.addConstr(
                S[i, j]
                + processing_times[i][j][k - resource_ind[i][0]]
                <= S[i_, j_] + big_M * (1 - x[i, i_, j, j_, k])
            )

        # Const. 4: Separation
        if (i == 0 or i == num_ops - 1) and (i_ == 0 or i_ == num_ops - 1):
            model.addConstr(
                S[i_, j_]
                >= S[i, j]
                + processing_times[i][j][k - resource_ind[i][0]]
                + ST[(i, i_)][j][j_][k]
                - big_M * (1 - x[i, i_, j, j_, k])
            )

    def _extract_solution(
            self,
            model: Model,
            instance: Instance,
            variables: Dict,
            solver_name: str,
            is_deadlock,
            is_runtime_over,
            is_rhc,
            **kwargs,
    ) -> Solution:
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
        sim_end_time = 0

        is_solution_exist = True
        is_infeasible = model.Status == GRB.INFEASIBLE or model.Status == GRB.INF_OR_UNBD

        if model.Status == GRB.TIME_LIMIT and model.SolCount == 0:
            is_runtime_over = True
            is_solution_exist = False
        elif is_infeasible:
            is_runtime_over = True
            is_solution_exist = False
        else:
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

        # extract option parameter for objective selection
        objective_option = kwargs.get('objective_option', "weighted_sum")

        return Solution(
            obj_val,
            runtime,
            sim_end_time,
            start_times,
            finish_times,
            assigned_resources,
            arrival_time_tardiness,
            departure_time_tardiness,
            resource_ind,
            solver_name,
            instance.objective_weights,
            instance,
            is_deadlock,
            is_runtime_over,
            is_solution_exist,
            is_infeasible,
            objective_option,
        )


class ExactSolver(SolverStrategy):
    """Exact solver (일반 + RHC 둘 다 처리)."""

    def solve(self, instance: Instance, is_numerical_exp: bool, **kwargs) -> Solution:
        # 🔹 RHC 관련 인자들을 kwargs에서 꺼내오기 (없을 수도 있음)
        processing_vehicles_op = kwargs.get("processing_vehicles_op")
        processing_vehicles_res = kwargs.get("processing_vehicles_res")
        horizon_start = kwargs.get("horizon_start")
        scheduler_runtime_limit = kwargs.get("scheduler_runtime_limit")

        # RHC 모드인지 여부 판단
        is_rhc = (
            processing_vehicles_op is not None
            and processing_vehicles_res is not None
            and horizon_start is not None
        )

        # 🔹 base model 만들 때 RHC 정보 같이 넘김
        model, variables = self._setup_base_model(
            instance,
            processing_vehicles_op=processing_vehicles_op if is_rhc else None,
            processing_vehicles_res=processing_vehicles_res if is_rhc else None,
        )

        # 🔹 objective 설정 (기존 Exact / ExactRHC에서 쓰던 로직 그대로)
        weights = instance.objective_weights
        num_vehicle = instance.num_vehicles
        T_a, T_d = variables["T_a"], variables["T_d"]
        obj_option = kwargs.get("obj_option", "weighted_sum")

        # weighted sum of arrival delay and departure delay
        if obj_option == "weighted_sum":
            model.setObjective(
                quicksum(
                    weights[0] * T_a[i] + weights[1] * T_d[i]
                    for i in range(num_vehicle)
                ),
                GRB.MINIMIZE,
            )

        # vehicle-wise max value of weighted sum of arrival delay and departure delay
        elif obj_option == "vehicle_wise_max":
            max_T_v_wise = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="max_T_v_wise")
            for i in range(num_vehicle):
                model.addConstr(max_T_v_wise >= weights[0] * T_a[i] + weights[1] * T_d[i])
            model.setObjectiveN(max_T_v_wise, index=0, priority=2)
            model.setObjectiveN(quicksum(weights[0] * T_a[i] + weights[1] * T_d[i] for i in range(num_vehicle)),
                                index=1, priority=1)

        # weighted sum of max value of arrival delay and departure delay
        elif obj_option == "weighted_sum_of_max":
            max_T_a = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="max_T_a")
            max_T_d = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="max_T_d")
            for i in range(num_vehicle):
                model.addConstr(max_T_a >= T_a[i])
                model.addConstr(max_T_d >= T_d[i])
            model.setObjectiveN(weights[0] * max_T_a + weights[1] * max_T_d, index=0, priority=2)
            model.setObjectiveN(quicksum(weights[0] * T_a[i] + weights[1] * T_d[i] for i in range(num_vehicle)),
                                index=1, priority=1)

        # # vehicle-wise max value of weighted sum of arrival delay and departure delay
        # elif obj_option == "vehicle_wise_max":
        #     lambda_ = 1e-04
        #     max_T_v_wise = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="max_T_v_wise")
        #     for i in range(num_vehicle):
        #         model.addConstr(max_T_v_wise >= weights[0] * T_a[i] + weights[1] * T_d[i])
        #     model.setObjective(
        #         max_T_v_wise + lambda_ * quicksum(
        #             weights[0] * T_a[i] + weights[1] * T_d[i] for i in range(num_vehicle)
        #         ),
        #         GRB.MINIMIZE,
        #     )
        #
        # # weighted sum of max value of arrival delay and departure delay
        # elif obj_option == "weighted_sum_of_max":
        #     lambda_ = 1e-04
        #     max_T_a = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="max_T_a")
        #     max_T_d = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="max_T_d")
        #     for i in range(num_vehicle):
        #         model.addConstr(max_T_a >= T_a[i])
        #         model.addConstr(max_T_d >= T_d[i])
        #     model.setObjective(
        #         weights[0] * max_T_a + weights[1] * max_T_d + lambda_ * quicksum(
        #             weights[0] * T_a[i] + weights[1] * T_d[i] for i in range(num_vehicle)
        #         ),
        #         GRB.MINIMIZE,
        #     )

        # 🔹 RHC 모드일 때만, 현재 처리 중인 작업을 horizon_start 에 고정
        resource_ind, landing_op_id, buffer_in_op_id, gate_op_id, buffer_out_op_id, takeoff_op_id = self._setup_resource_indices(instance)

        if is_rhc:
            S, y, x = variables["S"], variables["y"], variables["x"]
            for i, op_idx in enumerate(processing_vehicles_op):
                model.addConstr(y[op_idx, i, processing_vehicles_res[i]] == 1)
                model.addConstr(S[op_idx, i] == horizon_start)
                # if op_idx in [1, 3]:
                #     model.addConstr(S[op_idx, i] == horizon_start - 1e-06)      # buffer를 사용하지 않는 vehicle들이 시작지점에서 interupt 되는 문제를 방지하기 위해 이전부터 buffer를 점유하고 있었음을 의미하는 작은 마진을 줌
                # else:
                #     model.addConstr(S[op_idx, i] == horizon_start)

            # RHC 모드일 때, precedence 사전 선언 for buffer (due to zero processing time)
            for i, op_idx in enumerate(processing_vehicles_op):
                if op_idx not in [landing_op_id, gate_op_id]:
                    continue
                for i_, op_idx_ in enumerate(processing_vehicles_op):
                    if op_idx_ not in [buffer_in_op_id, buffer_out_op_id]:
                        continue
                    for r in range(resource_ind[op_idx + 1][0], resource_ind[op_idx + 1][1]):
                        model.addConstr(x[op_idx + 1, op_idx_, i, i_, r] == 0)


        # 🔹 Solve (RHC는 time limit 더 타이트하게)
        if is_numerical_exp or is_rhc:
            model.setParam("TimeLimit", scheduler_runtime_limit)
            model.optimize()
            if model.Status == GRB.TIME_LIMIT:
                is_runtime_over = True
            elif model.Status == GRB.OPTIMAL:
                is_runtime_over = False
            else:
                is_runtime_over = False  # 필요하면 추가 처리
        else:
            model.Params.MIPFocus = 1  # Prioritize finding better feasible solutions early
            model.optimize()
            is_runtime_over = False

        is_deadlock = False
        solver_name = "exact_RHC" if is_rhc else "exact"

        return self._extract_solution(
            model,
            instance,
            variables,
            solver_name,
            is_deadlock,
            is_runtime_over,
            is_rhc,
            objective_option=obj_option,
        )


class FCFSSolver(SolverStrategy):
    """First-Come-First-Served solver implementation."""

    def __init__(self, is_objective_enabled: bool = False, FCFS_for_landing_only: bool = False):
        self.use_gurobi = is_objective_enabled
        self.landing_only = FCFS_for_landing_only

    def solve(self, instance: Instance, is_numerical_exp: bool, **kwargs) -> Solution:
        model, variables = self._setup_base_model(instance)

        scheduler_runtime_limit = kwargs.get("scheduler_runtime_limit")
        obj_option = kwargs.get('obj_option', "weighted_sum")
        is_runtime_over = False

        # Set objective only for Gurobi variant
        if self.use_gurobi:
            weights = instance.objective_weights
            num_vehicle = instance.num_vehicles
            T_a = variables['T_a']
            T_d = variables['T_d']

            # extract option parameter for objective selection
            obj_option = kwargs.get('obj_option', "weighted_sum")

            # weighted sum of arrival delay and departure delay
            if obj_option == "weighted_sum":
                model.setObjective(quicksum(weights[0] * T_a[i] + weights[1] * T_d[i] for i in range(num_vehicle)),
                                   GRB.MINIMIZE)

            # vehicle-wise max value of weighted sum of arrival delay and departure delay
            elif obj_option == "vehicle_wise_max":
                max_T_v_wise = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="max_T_v_wise")
                for i in range(num_vehicle):
                    model.addConstr(max_T_v_wise >= weights[0] * T_a[i] + weights[1] * T_d[i])
                model.setObjectiveN(max_T_v_wise, index=0, priority=2)
                model.setObjectiveN(quicksum(weights[0] * T_a[i] + weights[1] * T_d[i] for i in range(num_vehicle)),
                                    index=1, priority=1)

            # weighted sum of max value of arrival delay and departure delay
            elif obj_option == "weighted_sum_of_max":
                max_T_a = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="max_T_a")
                max_T_d = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="max_T_d")
                for i in range(num_vehicle):
                    model.addConstr(max_T_a >= T_a[i])
                    model.addConstr(max_T_d >= T_d[i])
                model.setObjectiveN(weights[0] * max_T_a + weights[1] * max_T_d, index=0, priority=2)
                model.setObjectiveN(quicksum(weights[0] * T_a[i] + weights[1] * T_d[i] for i in range(num_vehicle)),
                                    index=1, priority=1)

        # Add FCFS constraints
        if self.landing_only:
            self._add_fcfs_landing_constraints(model, instance, variables)
            solver_name = "FCFS_landing_Gurobi" if self.use_gurobi else "FCFS_landing_SAT"
        else:
            self._add_fcfs_constraints(model, instance, variables)
            solver_name = "FCFS_Gurobi" if self.use_gurobi else "FCFS_SAT"

        # Solve the model
        if is_numerical_exp:
            model.setParam('TimeLimit', scheduler_runtime_limit)
            model.optimize()
            if model.Status == GRB.TIME_LIMIT:
                is_runtime_over = True
            elif model.Status == GRB.OPTIMAL:
                is_runtime_over = False
        else:
            model.optimize()
            is_runtime_over = False

        is_deadlock = False

        return self._extract_solution(model, instance, variables, solver_name, is_deadlock, is_runtime_over, objective_option=obj_option)

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

    def solve(self, instance: Instance, is_numerical_exp: bool, **kwargs) -> Solution:
        model, variables = self._setup_base_model(instance)
        is_runtime_over = False
        scheduler_runtime_limit = kwargs.get("scheduler_runtime_limit")

        # No objective function set for SAT mode
        if is_numerical_exp:
            model.setParam('TimeLimit', scheduler_runtime_limit)
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
    def solve(self, instance: Instance, is_numerical_exp: bool, **kwargs) -> Solution:
        # Create simulator
        simulator = VertiportSimulator(instance, is_numerical_exp)
        is_deadlock = False
        is_runtime_over = False

        # Step-by-step control with FCFS logic
        solve_start_time = time.time()
        while not simulator.is_simulation_complete():
            event = simulator.step_to_next_event()

            if event is None and simulator.is_simulation_complete() is False and is_numerical_exp:
                is_deadlock = True
                break

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

                            # Error fixing of separation time reflection; only the worst case separation is considered before
                            a = 0
                            if resource.state == ResourceState.SEPARATION_DELAY and event.event_type == EventType.RESOURCE_AVAILABLE and resource.id == event.resource_id:
                                for k in range(len(event.data['vehicle_operation_pairs'])):
                                    if vehicle.id == event.data['vehicle_operation_pairs'][k]['vehicle_id'] and operation == event.data['vehicle_operation_pairs'][k]['operation']:
                                        resource.state = ResourceState.IDLE
                                        a = 1
                                        break

                            if simulator.can_assign_vehicle_to_resource(vehicle.id, resource.id, operation):
                                simulator.assign_vehicle_to_resource_now(vehicle.id, resource.id, operation)
                                break_flag = True
                                break
                            elif a:
                                resource.state = ResourceState.SEPARATION_DELAY
                        if break_flag:
                            break

        solve_end_time = time.time()
        runtime = solve_end_time - solve_start_time

        return simulator._generate_solution(runtime, is_deadlock, is_runtime_over, solver_type="FCFS_heuristic")


class FCFS_HeuristicRandomSolver(SolverStrategy):
    def solve(self, instance: Instance, is_numerical_exp: bool, **kwargs) -> Solution:
        # Create simulator
        simulator = VertiportSimulator(instance, is_numerical_exp)
        is_deadlock = False
        is_runtime_over = False

        # Step-by-step control with FCFS logic
        solve_start_time = time.time()
        while not simulator.is_simulation_complete():
            event = simulator.step_to_next_event()

            if event is None and simulator.is_simulation_complete() is False and is_numerical_exp:
                is_deadlock = True
                break

            # FCFS scheduling logic here - TODO : improve more - multiple vehicles and multiple resources
            for operation in range(instance.num_operations):
                waiting_vehicles = simulator.get_waiting_vehicles(operation)
                available_resources = simulator.get_available_resources(operation)
                np.random.shuffle(available_resources)

                # Make assignment decisions
                if waiting_vehicles and available_resources:
                    break_flag = False
                    for i in range(len(waiting_vehicles)):
                        for j in range(len(available_resources)):
                            vehicle = waiting_vehicles[i]  # selection logic
                            resource = available_resources[j]  # selection logic

                            # Error fixing of separation time reflection; only the worst case separation is considered before
                            a = 0
                            if resource.state == ResourceState.SEPARATION_DELAY and event.event_type == EventType.RESOURCE_AVAILABLE and resource.id == event.resource_id:
                                for k in range(len(event.data['vehicle_operation_pairs'])):
                                    if vehicle.id == event.data['vehicle_operation_pairs'][k]['vehicle_id'] and operation == event.data['vehicle_operation_pairs'][k]['operation']:
                                        resource.state = ResourceState.IDLE
                                        a = 1
                                        break

                            if simulator.can_assign_vehicle_to_resource(vehicle.id, resource.id, operation):
                                simulator.assign_vehicle_to_resource_now(vehicle.id, resource.id, operation)
                                break_flag = True
                                break
                            elif a:
                                resource.state = ResourceState.SEPARATION_DELAY
                        if break_flag:
                            break

        solve_end_time = time.time()
        runtime = solve_end_time - solve_start_time

        return simulator._generate_solution(runtime, is_deadlock, is_runtime_over, solver_type="FCFS_heuristic_random")


class RunRHC(SolverStrategy):
    def solve(self, instance: Instance, is_numerical_exp: bool, **kwargs) -> Solution:
        planned_resource_assignment = kwargs["planned_resource_assignment"]
        planned_operation_start_times = kwargs["planned_operation_start_times"]
        vehicle_original_id = kwargs["vehicle_original_id"]
        obj_option = kwargs.get("obj_option", "weighted_sum")
        next_current_time = kwargs["next_current_time"]
        past_event_time = 0.0

        resource_ind, landing_op_id, buffer_in_op_id, gate_op_id, buffer_out_op_id, takeoff_op_id = self._setup_resource_indices(instance)

        # Create simulator
        simulator = VertiportSimulatorRecedingHorizon(instance, is_numerical_exp)
        is_deadlock = False
        is_runtime_over = False

        # Step-by-step control with RunRHC logic
        solve_start_time = time.time()
        while not simulator.is_simulation_complete():
            event = simulator.step_to_next_event()

            if event is None and simulator.is_simulation_complete() is False and is_numerical_exp and past_event_time >= next_current_time:
                is_deadlock = True
                break

            # RunRHC logic here - TODO : improve more - multiple vehicles and multiple resources
            break_flag = False
            for operation in range(instance.num_operations):
                waiting_vehicles = simulator.get_waiting_vehicles(operation)
                available_resources = simulator.get_available_resources(operation)

                # Make assignment decisions
                if waiting_vehicles and available_resources:
                    break_flag2 = False
                    for i in range(len(waiting_vehicles)):
                        for j in range(len(available_resources)):
                            vehicle = waiting_vehicles[i]  # selection logic
                            resource = available_resources[j]  # selection logic
                            if resource.id == planned_resource_assignment[vehicle_original_id[vehicle.id], operation]\
                                    or (operation in [buffer_in_op_id, buffer_out_op_id] and (round(planned_operation_start_times[vehicle_original_id[vehicle.id], operation + 1], 9)
                                        == round(planned_operation_start_times[vehicle_original_id[vehicle.id], operation], 9) or round(simulator.current_time, 9) >= round(planned_operation_start_times[vehicle_original_id[vehicle.id], operation + 1], 9))):
                                if round(simulator.current_time, 9) >= round(planned_operation_start_times[vehicle_original_id[vehicle.id], operation], 9):

                                    if len(resource.log_operation_finish_times) > 0 and operation in [landing_op_id, takeoff_op_id]:
                                        finish = resource.log_operation_finish_times[-1]
                                        pre_v = resource.previous_vehicle_operation[0]
                                        pre_op = resource.previous_vehicle_operation[1]
                                        available_time = finish + instance.ST[(pre_op, operation)][pre_v][vehicle.id][resource.id]
                                        if round(available_time, 9) > round(event.time, 9):
                                            if resource.state == ResourceState.IDLE:
                                                resource.state = ResourceState.SEPARATION_DELAY
                                        else:
                                            if resource.state != ResourceState.PROCESSING:
                                                resource.state = ResourceState.IDLE

                                    # a = 0
                                    # # if resource.state == ResourceState.SEPARATION_DELAY and (event.event_type == EventType.RESOURCE_AVAILABLE or event.event_type == EventType.VEHICLE_ARRIVAL or event.event_type == EventType.OPERATION_COMPLETE) and resource.id == event.resource_id:
                                    # if resource.state == ResourceState.SEPARATION_DELAY and event.event_type == EventType.RESOURCE_AVAILABLE and resource.id == event.resource_id:
                                    #     if event.data != {}:
                                    #         for k in range(len(event.data['vehicle_operation_pairs'])):
                                    #             if vehicle.id == event.data['vehicle_operation_pairs'][k][
                                    #                 'vehicle_id'] and operation == event.data['vehicle_operation_pairs'][k]['operation']:
                                    #                 resource.state = ResourceState.IDLE
                                    #                 a = 1
                                    #                 break

                                    if simulator.can_assign_vehicle_to_resource(vehicle.id, resource.id, operation):
                                        simulator.assign_vehicle_to_resource_now(vehicle.id, resource.id, operation)
                                        if operation in [gate_op_id, takeoff_op_id]:
                                            simulator.resources[vehicle.log_assigned_resources[-1]].state = ResourceState.IDLE
                                        break_flag = True
                                        break_flag2 = True
                                        break
                                    # elif a:
                                    #     resource.state = ResourceState.SEPARATION_DELAY
                                    elif (event.event_type == EventType.RESOURCE_AVAILABLE or event.event_type == EventType.VEHICLE_ARRIVAL or event.event_type == EventType.OPERATION_COMPLETE) and resource.state == ResourceState.SEPARATION_DELAY:
                                        available_time = 0.0
                                        if len(resource.log_operation_finish_times) > 0 and operation in [landing_op_id, takeoff_op_id]:
                                            finish = resource.log_operation_finish_times[-1]
                                            pre_v = resource.previous_vehicle_operation[0]
                                            pre_op = resource.previous_vehicle_operation[1]
                                            available_time = finish + instance.ST[(pre_op, operation)][pre_v][vehicle.id][resource.id]
                                        new_event_time = max(simulator.current_time, available_time)
                                        separation_clear_event = Event(time=new_event_time,
                                                                       event_type=EventType.RESOURCE_AVAILABLE,
                                                                       vehicle_id=vehicle.id,
                                                                       resource_id=resource.id,
                                                                       operation_id=operation,
                                                                       event_id=event.event_id)
                                        heapq.heappush(simulator.event_queue, separation_clear_event)
                                        resource.state = ResourceState.IDLE
                                else:
                                    if operation == 0 and event.event_type == EventType.VEHICLE_ARRIVAL:
                                        available_time = 0.0
                                        if len(resource.log_operation_finish_times) > 0 and operation in [landing_op_id, takeoff_op_id]:
                                            finish = resource.log_operation_finish_times[-1]
                                            pre_v = resource.previous_vehicle_operation[0]
                                            pre_op = resource.previous_vehicle_operation[1]
                                            available_time = finish + instance.ST[(pre_op, operation)][pre_v][vehicle.id][resource.id]
                                        new_event_time = max(planned_operation_start_times[vehicle_original_id[vehicle.id], operation],
                                                             available_time)
                                        arrival_event = Event(time=new_event_time,
                                                              event_type=EventType.RESOURCE_AVAILABLE,
                                                              vehicle_id=vehicle.id,
                                                              resource_id=resource.id,
                                                              operation_id=operation,
                                                              event_id=event.event_id)
                                        heapq.heappush(simulator.event_queue, arrival_event)
                                        vehicle.state = VehicleState.WAITING_FOR_LANDING
                                    elif event.event_type == EventType.OPERATION_COMPLETE:
                                        available_time = 0.0
                                        if len(resource.log_operation_finish_times) > 0 and operation in [landing_op_id, takeoff_op_id]:
                                            finish = resource.log_operation_finish_times[-1]
                                            pre_v = resource.previous_vehicle_operation[0]
                                            pre_op = resource.previous_vehicle_operation[1]
                                            available_time = finish + instance.ST[(pre_op, operation)][pre_v][vehicle.id][resource.id]
                                        new_event_time = max(planned_operation_start_times[vehicle_original_id[vehicle.id], operation],
                                                             available_time)
                                        completion_event = Event(time=new_event_time,
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
                        if break_flag2:
                            break
                    if break_flag:
                        break

            past_event_time = event.time

        solve_end_time = time.time()
        runtime = solve_end_time - solve_start_time

        return simulator._generate_solution(runtime, is_deadlock, is_runtime_over, solver_type="run_RHC", obj_option=obj_option, next_current_time=next_current_time)


# Solver factory with minimal overhead
_SOLVER_INSTANCES = {
    "exact": ExactSolver(),
    "FCFS_SAT": FCFSSolver(is_objective_enabled=False, FCFS_for_landing_only=False),
    "FCFS_Gurobi": FCFSSolver(is_objective_enabled=True, FCFS_for_landing_only=False),
    "FCFS_landing_SAT": FCFSSolver(is_objective_enabled=False, FCFS_for_landing_only=True),
    "FCFS_landing_Gurobi": FCFSSolver(is_objective_enabled=True, FCFS_for_landing_only=True),
    "no_rule_SAT": NoRuleSolver(),
    "FCFS_heuristic": FCFS_HeuristicSolver(),
    "FCFS_heuristic_random": FCFS_HeuristicRandomSolver(),
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
                Available options: "exact", "FCFS_SAT", "FCFS_Gurobi",
                                   "FCFS_landing_SAT", "FCFS_landing_Gurobi",
                                   "no_rule_SAT",
                                   "FCFS_heuristic", "FCFS_heuristic_random",
                                   "run_RHC",
    
    Returns:
        Solution: Solution object containing optimization results
        
    Raises:
        ValueError: If solver type is not recognized
    """
    if solver not in _SOLVER_INSTANCES:
        available_solvers = ", ".join(_SOLVER_INSTANCES.keys())
        raise ValueError(f"Unknown solver type: '{solver}'. Available solvers: {available_solvers}")
    
    return _SOLVER_INSTANCES[solver].solve(instance, is_numerical_exp, **kwargs)
