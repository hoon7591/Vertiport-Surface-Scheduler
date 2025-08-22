"""
Discrete Event Simulation Engine for Vertiport Surface Scheduling

This module provides a general-purpose simulation engine for modeling vertiport operations.
The engine maintains all simulation state and provides methods for resource allocation
and simulation control that can be used by external scheduling algorithms.
"""

import heapq
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from collections import defaultdict

from Instance import Instance
from Solution import Solution


class EventType(Enum):
    """Types of simulation events"""
    VEHICLE_ARRIVAL = auto()
    OPERATION_START = auto()
    OPERATION_COMPLETE = auto()
    RESOURCE_AVAILABLE = auto()  # Unified event for resource availability (replaces both old RESOURCE_AVAILABLE and SEPARATION_TIME_UPDATE)
    PLANNED_DEPARTURE_REACHED = auto()
    VEHICLE_DEPARTURE = auto()
    SIMULATION_END = auto()


class VehicleState(Enum): # TODO : some of them are useless for our setting
    """States of vehicles in the simulation"""
    BEFORE_ARRIVAL = auto()      # Not yet arrived at vertiport
    WAITING_FOR_LANDING = auto()  # Waiting for landing pad
    LANDING = auto()          # Currently landing
    WAITING_AFTER_LANDING = auto() # Waiting for buffer-in
    IN_BUFFER_IN = auto()     # In buffer-in area
    WAITING_FOR_GATE_AT_BUFFER = auto()     # Waiting for gate at buffer
    AT_GATE = auto()          # At gate (charging/passenger ops)
    WAITING_AFTER_GATE = auto() # Waiting for buffer-out
    IN_BUFFER_OUT = auto()    # In buffer-out area
    WAITING_FOR_TAKEOFF_AT_BUFFER = auto()  # Waiting for takeoff
    TAKING_OFF = auto()       # Currently taking off
    DEPARTED = auto()         # Left vertiport

class VehicleStateToOperation:
    """Mapping from vehicle states to operation indices"""
    mapping = {
        VehicleState.BEFORE_ARRIVAL: -1,
        VehicleState.WAITING_FOR_LANDING: 0,
        VehicleState.LANDING: 0,
        VehicleState.WAITING_AFTER_LANDING: 1,
        VehicleState.IN_BUFFER_IN: 1,
        VehicleState.WAITING_FOR_GATE_AT_BUFFER: 2,
        VehicleState.AT_GATE: 2,
        VehicleState.WAITING_AFTER_GATE: 3,
        VehicleState.IN_BUFFER_OUT: 3,
        VehicleState.WAITING_FOR_TAKEOFF_AT_BUFFER: 4,
        VehicleState.TAKING_OFF: 4,
        VehicleState.DEPARTED: 9999
    }
    
    @classmethod
    def get(cls, state: VehicleState, default=-404):
        """Get operation index for a vehicle state"""
        return cls.mapping.get(state, default)

class ResourceState(Enum): # TODO : check, separation_delay depends on vehicle type sequences. is it possible to store?
    """States of resources in the simulation"""
    IDLE = auto()
    PROCESSING = auto()
    OCCUPIED = auto()
    SEPARATION_DELAY = auto()  # In separation time after previous operation


@dataclass
class Event:
    """Simulation event"""
    time: float
    event_type: EventType
    vehicle_id: int
    resource_id: Optional[int] = None
    operation_id: Optional[int] = None
    event_id: Optional[int] = None  # Unique ID for tracking and deletion
    data: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For heapq ordering"""
        if self.time != other.time:
            return self.time < other.time
        return self.event_type.value < other.event_type.value


@dataclass
class Vehicle:
    """Vehicle representation in simulation"""
    id: int
    vehicle_type: int
    arrival_time: float
    planned_arrival_time: float
    planned_departure_time: float
    planned_gate_close_time: float
    state: VehicleState = VehicleState.BEFORE_ARRIVAL
    current_operation: int = -1
    current_resource: Optional[int] = None
    log_start_times: List[float] = field(default_factory=list)
    log_operation_finish_times: List[float] = field(default_factory=list)
    log_assigned_resources: List[int] = field(default_factory=list)
    waiting_duration: float = 0.0
    operation_duration: float = 0.0

    def update_durations(self, prev_time, current_time) -> None:
        """Update vehicle's waiting and operation durations based on time elapsed"""
        duration = current_time - prev_time
        if self.state == (VehicleState.WAITING_FOR_LANDING or VehicleState.WAITING_AFTER_LANDING or VehicleState.WAITING_AFTER_GATE or VehicleState.WAITING_FOR_TAKEOFF_AT_BUFFER):
            self.waiting_duration += duration
        self.operation_duration += duration


@dataclass
class Resource:
    """Resource representation in simulation"""
    id: int
    resource_type: str  # 'pad', 'buffer_in', 'gate', 'buffer_out'
    state: ResourceState = ResourceState.IDLE
    current_vehicle: Optional[int] = None
    previous_vehicle_operation: Optional[Tuple[int, int]] = None  # (vehicle_id, operation_id) For separation time calculations
    utilization_duration: float = 0.0 # How much time utilized while simulating
    idle_duration: float = 0.0 # How much time IDLEed while simulating
    occupied_duration: float = 0.0 # How much time occupied by any vehicle while simulating

    # Dynamic separation time tracking
    allocation_prohibited_vehicle_n_operation: List[Tuple[int, int]] = field(default_factory=list)  # Vehicle IDs and their operations that can be allocated
    separation_time_reached_events: List[int] = field(default_factory=list)  # Event IDs for separation updates
    last_separation_update_time: float = 0.0  # Last time allocatable vehicles were updated
    
    # Historical tracking (parallel to Vehicle object tracking)
    log_operation_start_times: List[float] = field(default_factory=list)  # When operations started on this resource
    log_operation_finish_times: List[float] = field(default_factory=list)  # When operations finished on this resource
    log_allocated_vehicles: List[int] = field(default_factory=list)  # Vehicle IDs that used this resource (in chronological order)
    log_operation_types: List[int] = field(default_factory=list)  # Operation IDs that were performed on this resource

    def update_durations(self, prev_time, current_time) -> None:
        """Update resource's waiting and operation durations based on time elapsed"""
        duration = current_time - prev_time
        if self.state == ResourceState.IDLE:
            self.idle_duration += duration
        elif self.state == ResourceState.OCCUPIED:
            self.occupied_duration += duration
        elif self.state == ResourceState.PROCESSING:
            self.utilization_duration += duration


class VertiportSimulator:
    """
    Discrete event simulation engine for vertiport operations.
    
    This engine maintains all simulation state and provides methods for:
    - Resource allocation decisions
    - Simulation control (step-by-step or time-based execution)
    - State inspection and statistics collection
    
    External scheduling algorithms can use this engine by calling resource
    allocation methods and controlling simulation execution.
    """
    
    def __init__(self, instance: Instance):
        self.instance = instance
        
        # Simulation state
        self.current_time = 0.0
        self.event_queue: List[Event] = []
        self.vehicles: Dict[int, Vehicle] = {}
        self.resources: Dict[int, Resource] = {}
        
        # Event management
        self.next_event_id = 1  # Counter for unique event IDs
        
        # Current State
        self.stats = {
            'total_vehicles': 0,
            'completed_vehicles': 0,
            'total_waiting_time': 0.0,
            'total_flow_time': 0.0,
            'resource_utilization': {},
            'queue_lengths': defaultdict(list),
            'service_times': defaultdict(list),
            'tardiness': {'arrival': [], 'departure': []},
            'events_processed': 0
        }
        
        # Waiting queues for each operation type
        self.waiting_vehicles_for_each_operation: Dict[int, List[Vehicle]] = defaultdict(list)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Configure logging to save to files if not already configured
        if not self.logger.handlers:
            # Set logger level
            self.logger.setLevel(logging.DEBUG)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Create file handler for detailed logs
            file_handler = logging.FileHandler('vertiport_simulation.log', mode='w')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            
            # Create file handler for errors only
            error_handler = logging.FileHandler('vertiport_errors.log', mode='w')
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            
            # Create console handler for important messages
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            
            # Add handlers to logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(error_handler)
            self.logger.addHandler(console_handler)
            
            self.logger.info("Vertiport Simulator logging initialized - logs saved to files")
        
        self._initialize_simulation()
        
    def _initialize_simulation(self):
        """Initialize simulation components"""
        
        # Create vehicles
        for v_id in range(self.instance.num_vehicles):
            vehicle = Vehicle(
                id=v_id,
                vehicle_type=self.instance.vehicle_type[v_id],
                arrival_time=self.instance.vehicle_arrival_times[v_id],
                planned_arrival_time=self.instance.vehicle_planned_arrival_times[v_id],
                planned_departure_time=self.instance.vehicle_planned_departure_times[v_id],
                planned_gate_close_time=self.instance.vehicle_planned_gate_close_times[v_id]
            )
            self.vehicles[v_id] = vehicle
            
            # Schedule arrival event
            arrival_event = Event(
                time=vehicle.arrival_time,
                event_type=EventType.VEHICLE_ARRIVAL,
                vehicle_id=v_id,
                event_id=self._get_next_event_id()
            )
            heapq.heappush(self.event_queue, arrival_event)

            # Schdule ETD reached event
            departure_event = Event(
                time = vehicle.planned_departure_time,
                event_type = EventType.PLANNED_DEPARTURE_REACHED,
                vehicle_id = v_id,
                event_id = self._get_next_event_id()
            )
            heapq.heappush(self.event_queue, departure_event)

        # Create resources
        self._create_resources()
        
        # Initialize statistics
        self.stats['total_vehicles'] = len(self.vehicles)
        for res_id in self.resources:
            self.stats['resource_utilization'][res_id] = 0.0
    
    # ========================= SIMULATION CONTROL METHODS =========================
    
    def step_to_next_event(self) -> Optional[Event]:
        """
        Execute simulation until the next event and return the event.
        Returns None if no more events or simulation complete.
        """
        if not self.event_queue:
            return None
        
        # Get next event
        event = heapq.heappop(self.event_queue)
        if event.time < self.current_time:
            raise RuntimeError("Event time cannot be in the past")
        
        # time is tickin & save duration information to every vehicles and resources
        for vehicle in self.vehicles.values():
            vehicle.update_durations(self.current_time, event.time)
        for resource in self.resources.values():
            resource.update_durations(self.current_time, event.time)

        self.current_time = event.time
        # Process event
        self._process_event(event)
        self.stats['events_processed'] += 1
        
        # Record queue lengths
        for op_id, queue in self.waiting_vehicles_for_each_operation.items():
            self.stats['queue_lengths'][op_id].append(len(queue))
        
        return event
    
    def run_simulation_until(self, max_time: float) -> List[Event]:
        """
        Run simulation until specified time.
        Returns list of events processed.
        """
        processed_events = []
        
        while self.event_queue and self.current_time < max_time:
            event = self.step_to_next_event()
            if event is None:
                break
            processed_events.append(event)
            
            # Check if all vehicles completed
            if self.stats['completed_vehicles'] >= self.stats['total_vehicles']:
                break
        
        return processed_events
    
    # ========================= INTERNAL EVENT HANDLING METHODS =========================
        
    def run_complete_simulation(self, max_time: Optional[float] = None) -> Solution:
        """Run simulation to completion and return Solution object"""
        if max_time is None:
            max_time = max(self.instance.vehicle_planned_arrival_times) + 1000  # Safety margin
        
        self.logger.info("Starting simulation with %d vehicles", len(self.vehicles))
        
        processed_events = self.run_simulation_until(max_time)
        
        self.logger.info("Simulation completed. Processed %d events", len(processed_events))
        
        return self._generate_solution()
    
    # ========================= RESOURCE ALLOCATION METHODS =========================
    
    def get_waiting_vehicles(self, operation: int) -> List[Vehicle]:
        """Get list of vehicles waiting for specified operation"""
        return self.waiting_vehicles_for_each_operation[operation].copy()
    
    def _get_next_event_id(self) -> int:
        """Generate unique event ID"""
        event_id = self.next_event_id
        self.next_event_id += 1
        return event_id
    
    def _clear_separation_events_for_resource(self, resource_id: int) -> None:
        """Clear all pending separation time events for a specific resource"""
        if resource_id not in self.resources:
            return
        
        resource = self.resources[resource_id]
        
        # Remove events from queue
        events_to_remove = set(resource.separation_time_reached_events)
        if events_to_remove:
            # Filter out events to remove and rebuild heap
            filtered_events = [event for event in self.event_queue 
                             if event.event_id not in events_to_remove]
            self.event_queue = filtered_events
            heapq.heapify(self.event_queue)
        
        # Clear pending events list
        resource.separation_time_reached_events.clear()

    def preview_assignment_impact(self, vehicle_id: int, resource_id: int, operation: int) -> Dict[str, Any]:
        """
        Preview the impact of assigning a vehicle to a resource, including separation time effects.
        This helps external schedulers make informed decisions.
        
        Returns:
        - start_time: When the operation would start
        - finish_time: When the operation would finish  
        - separation_time: Required separation time after completion
        - resource_available_time: When resource becomes available for next use
        """
        if (vehicle_id not in self.vehicles or resource_id not in self.resources):
            return {}
        
        vehicle = self.vehicles[vehicle_id]
        resource = self.resources[resource_id]
        
        # Calculate when operation can start
        start_time = max(self.current_time, resource.idle_duration)
        
        # Get processing time
        processing_time = self.get_vehicle_processing_time(vehicle_id, resource_id, operation)
        finish_time = start_time + processing_time
        
        # Calculate conservative separation time
        separation_time = self._calculate_conservative_separation_time(
            resource, operation, vehicle, resource_id
        )
        
        # When resource becomes available for next use
        resource_available_time = finish_time + separation_time
        
        return {
            'start_time': start_time,
            'finish_time': finish_time,
            'processing_time': processing_time,
            'separation_time': separation_time,
            'resource_available_time': resource_available_time,
            'waiting_time': start_time - self.current_time
        }

    def get_available_resources(self, operation: int) -> List[Resource]:
        """Get list of resources available for specified operation"""
        resource_ranges = self._build_resource_ranges()
        if operation >= len(resource_ranges):
            return []
        
        start_idx, end_idx = resource_ranges[operation]
        available = []
        
        for res_id in range(start_idx, end_idx):
            resource = self.resources[res_id]
            if resource.state == ResourceState.IDLE:
                available.append(resource)
        
        return available
    
    def can_assign_vehicle_to_resource(self, vehicle_id: int, resource_id: int, operation: int) -> bool:
        """Check if a vehicle can be assigned to a specific resource for an operation"""
        if vehicle_id not in self.vehicles or resource_id not in self.resources:
            return False
        
        vehicle = self.vehicles[vehicle_id]
        resource = self.resources[resource_id]
        
        # Check if vehicle is waiting for this operation
        if vehicle.current_operation != operation:
            return False
        
        # Check if resource is available
        if resource.state != ResourceState.IDLE:
            return False
        
        # Check if processing time is available
        resource_ranges = self._build_resource_ranges()
        if operation >= len(resource_ranges):
            return False
        
        local_idx = resource_id - resource_ranges[operation][0]
        if (local_idx < 0 or local_idx >= len(self.instance.proc[operation][vehicle_id])):
            return False

        # Check if early departure
        expected_operation = (len(resource_ranges) - 2) if self.instance.num_buffer_out > 0 \
            else (len(resource_ranges) - 1)
        if operation == expected_operation and self.current_time < vehicle.planned_gate_close_time:
            return False

        processing_time = self.instance.proc[operation][vehicle_id][local_idx]
        return processing_time >= 0
    
    def assign_vehicle_to_resource_at_future_time(self, vehicle_id: int, resource_id: int, operation: int, future_time: float) -> bool:
        """
        Assign a vehicle to a resource for an operation.
        Returns True if assignment successful, False otherwise.
        """
        if not self.can_assign_vehicle_to_resource(vehicle_id, resource_id, operation):
            return False

        vehicle = self.vehicles[vehicle_id]

        # Calculate when operation can start
        start_time = future_time
        
        # Schedule operation start
        start_event = Event(
            time=start_time,
            event_type=EventType.OPERATION_START,
            vehicle_id=vehicle_id,
            resource_id=resource_id,
            operation_id=operation
        )
        heapq.heappush(self.event_queue, start_event)

        # get processing time for vehicle, resource combination
                
        return True
    
    def assign_vehicle_to_resource_now(self, vehicle_id: int, resource_id: int, operation: int) -> bool: # TODO : allocate resource to the vehicle at the future time
        """
        Assign a vehicle to a resource for an operation.
        Returns True if assignment successful, False otherwise.
        """
        
        return self.assign_vehicle_to_resource_at_future_time(vehicle_id, resource_id, operation, self.current_time)


    def get_vehicle_processing_time(self, vehicle_id: int, resource_id: int, operation: int) -> float:
        """Get processing time for a vehicle on a specific resource for an operation"""
        resource_ranges = self._build_resource_ranges()
        if operation >= len(resource_ranges):
            return 0.0
        
        local_idx = resource_id - resource_ranges[operation][0]
        if (local_idx < 0 or local_idx >= len(self.instance.proc[operation][vehicle_id])):
            return 0.0
        
        return self.instance.proc[operation][vehicle_id][local_idx]
    
    # ========================= STATE INSPECTION METHODS =========================
    
    def get_current_time(self) -> float:
        """Get current simulation time"""
        return self.current_time
    
    def get_next_event_time(self) -> Optional[float]:
        """Get time of next scheduled event, None if no events"""
        return self.event_queue[0].time if self.event_queue else None
    
    def get_vehicle_state(self, vehicle_id: int) -> Optional[Vehicle]:
        """Get current state of a vehicle"""
        return self.vehicles.get(vehicle_id)
    
    def get_resource_state(self, resource_id: int) -> Optional[Resource]:
        """Get current state of a resource"""
        return self.resources.get(resource_id)
    
    def get_queue_lengths(self) -> Dict[int, int]:
        """Get current queue lengths for each operation"""
        return {op_id: len(queue) for op_id, queue in self.waiting_vehicles_for_each_operation.items()}
    
    def is_simulation_complete(self) -> bool:
        """Check if simulation is complete (all vehicles departed)"""
        return self.stats['completed_vehicles'] >= self.stats['total_vehicles']
    
    def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive simulation statistics"""
        total_time = max(self.instance.vehicle_planned_arrival_times) if self.vehicles else 1.0
        
        # Calculate resource utilization
        for res_id, resource in self.resources.items():
            if total_time > 0:
                self.stats['resource_utilization'][res_id] = (
                    resource.utilization_duration / total_time * 100
                )
        
        # Calculate averages
        if self.stats['completed_vehicles'] > 0:
            self.stats['avg_waiting_time'] = (
                self.stats['total_waiting_time'] / self.stats['completed_vehicles']
            )
            self.stats['avg_flow_time'] = (
                self.stats['total_flow_time'] / self.stats['completed_vehicles']
            )
            if self.stats['tardiness']['arrival']:
                self.stats['avg_arrival_tardiness'] = np.mean(self.stats['tardiness']['arrival'])
            if self.stats['tardiness']['departure']:
                self.stats['avg_departure_tardiness'] = np.mean(self.stats['tardiness']['departure'])
        
        return self.stats
    
    def _create_resources(self):
        """Create simulation resources based on instance configuration"""
        resource_id = 0
        
        # Landing/Takeoff pads
        for _ in range(self.instance.num_pad):
            self.resources[resource_id] = Resource(
                id=resource_id,
                resource_type='pad'
            )
            resource_id += 1
        
        # Buffer-in areas
        for _ in range(self.instance.num_buffer_in):
            self.resources[resource_id] = Resource(
                id=resource_id,
                resource_type='buffer_in'
            )
            resource_id += 1
        
        # Gates
        for _ in range(self.instance.num_gate):
            self.resources[resource_id] = Resource(
                id=resource_id,
                resource_type='gate'
            )
            resource_id += 1
        
        # Buffer-out areas (if not unified)
        if not self.instance.is_unified_buffer:
            for _ in range(self.instance.num_buffer_out):
                self.resources[resource_id] = Resource(
                    id=resource_id,
                    resource_type='buffer_out'
                )
                resource_id += 1
    
    # ========================= INTERNAL EVENT HANDLING METHODS =========================
        
    def _process_event(self, event: Event):
        """Process a single simulation event"""
        if event.event_type == EventType.VEHICLE_ARRIVAL:
            self._handle_vehicle_arrival(event)
        elif event.event_type == EventType.OPERATION_START:
            self._handle_operation_start(event)
        elif event.event_type == EventType.OPERATION_COMPLETE:
            self._handle_operation_complete(event)
        elif event.event_type == EventType.RESOURCE_AVAILABLE:
            self._handle_resource_available(event)
        elif event.event_type == EventType.PLANNED_DEPARTURE_REACHED:
            self._handle_planned_departure_reached(event)
        elif event.event_type == EventType.VEHICLE_DEPARTURE:
            self._handle_vehicle_departure(event)
    
    def _handle_vehicle_arrival(self, event: Event):
        """Handle aircraft arrival at vertiport"""
        vehicle = self.vehicles[event.vehicle_id]
        vehicle.state = VehicleState.WAITING_FOR_LANDING
        vehicle.current_operation = VehicleStateToOperation.get(vehicle.state, default=0)
        
        self.logger.debug("Vehicle %d arrived at time %.2f", vehicle.id, self.current_time)
        
        # Add to landing queue
        self.waiting_vehicles_for_each_operation[0].append(vehicle)
    
    def _handle_operation_start(self, event: Event):
        """Handle start of an operation"""
        if event.resource_id is None or event.operation_id is None:
            return
            
        vehicle = self.vehicles[event.vehicle_id]
        resource = self.resources[event.resource_id]

        if resource.state != ResourceState.IDLE:
            return
    
        # clear the vehicle's previous resource occupancy
        if vehicle.current_resource is not None:
            prev_resource = self.resources[vehicle.current_resource]
            if prev_resource.separation_time_reached_events:
                prev_resource.state = ResourceState.SEPARATION_DELAY
            elif prev_resource.current_vehicle == vehicle.id:
                prev_resource.state = ResourceState.IDLE
                prev_resource.current_vehicle = None
            vehicle.current_resource = None

        operation = event.operation_id
        
        # Record start logs for vehicle
        vehicle.log_start_times.append(self.current_time)
        vehicle.log_assigned_resources.append(resource.id)

        # Record start logs for resource
        resource.log_operation_start_times.append(self.current_time)
        resource.log_allocated_vehicles.append(vehicle.id)
        resource.log_operation_types.append(operation)
        
        # Update vehicle state
        vehicle.current_resource = resource.id
        vehicle.state = self._get_operation_state(operation, in_progress=True)
        # Remove vehicle from waiting queue
        if vehicle in self.waiting_vehicles_for_each_operation[operation]:
            self.waiting_vehicles_for_each_operation[operation].remove(vehicle)


        vehicle.current_operation = VehicleStateToOperation.get(vehicle.state, default=operation)

        # Update resource state
        
            # clean pad separation related information
        if (resource.state == ResourceState.SEPARATION_DELAY or resource.allocation_prohibited_vehicle_n_operation):
            resource.allocation_prohibited_vehicle_n_operation.clear()
            self._clear_separation_events_for_resource(resource.id)

            # resource states update
        resource.state = ResourceState.PROCESSING
        resource.current_vehicle = vehicle.id
        
        # Calculate processing time
        resource_ranges = self._build_resource_ranges()
        local_idx = resource.id - resource_ranges[operation][0]
        processing_time = self.instance.proc[operation][vehicle.id][local_idx]
        
        # Schedule operation completion
        completion_event = Event(
            time=self.current_time + processing_time,
            event_type=EventType.OPERATION_COMPLETE,
            vehicle_id=vehicle.id,
            resource_id=resource.id,
            operation_id=operation,
            event_id=self._get_next_event_id()
        )
        heapq.heappush(self.event_queue, completion_event)
        
        self.logger.debug("Vehicle %d started operation %d on resource %d at time %.2f", 
                          vehicle.id, operation, resource.id, self.current_time)
    
    def _handle_operation_complete(self, event: Event):
        """Handle completion of an operation"""
        if event.resource_id is None or event.operation_id is None:
            return
            
        vehicle = self.vehicles[event.vehicle_id]
        resource = self.resources[event.resource_id]
        operation = event.operation_id
        
        # Record completion time for vehicle
        vehicle.log_operation_finish_times.append(self.current_time)
        
        # Record completion time for resource (parallel tracking)
        resource.log_operation_finish_times.append(self.current_time)
        
        # Update resource state for separation time
        resource.previous_vehicle_operation = (vehicle.id, operation)
        
        # Update vehicle state
        state = vehicle.state
        vehicle.state = VehicleState(state.value + 1)  # Move to next state
        if state == VehicleState.TAKING_OFF:
            vehicle.state = VehicleState.DEPARTED
            # Vehicle completed all operations
            self._handle_vehicle_departure(Event(
                time=self.current_time,
                event_type=EventType.VEHICLE_DEPARTURE,
                vehicle_id=vehicle.id,
                event_id=self._get_next_event_id()
            ))

        vehicle.current_operation = VehicleStateToOperation.get(vehicle.state, default=operation + 1)
        if vehicle.current_operation is not None:
            self.waiting_vehicles_for_each_operation[vehicle.current_operation].append(vehicle)

        resource.state = ResourceState.SEPARATION_DELAY if state == VehicleState.TAKING_OFF else ResourceState.OCCUPIED

        self.logger.debug("Vehicle %d completed operation %d on resource %d at time %.2f",
                    vehicle.id, operation, resource.id, self.current_time)

        # creating separation time events
        list_of_separation_time = self._calculate_list_of_separation_time(resource, operation, vehicle)
        for separation_info in list_of_separation_time:
            self.logger.debug("Scheduled resource available event due to separation time required for resource %d at time %.2f: %s",
                              resource.id, self.current_time, separation_info)
            
            event_id = self._get_next_event_id()
            available_event = Event(
                time=separation_info['available_time'],
                event_type=EventType.RESOURCE_AVAILABLE,
                vehicle_id=-1,  # No specific vehicle for resource events
                resource_id=resource.id,
                event_id=event_id,
                data={
                    'event_subtype': 'separation_update',
                    'separation_time': separation_info['separation_time'],
                    'vehicle_operation_pairs': separation_info.get('vehicle_operation_pairs', [])
                }
            )
            heapq.heappush(self.event_queue, available_event)
            resource.separation_time_reached_events.append(event_id)  # Store event ID, not the event object
            # Update resource allocation prohibited vehicles
            resource.allocation_prohibited_vehicle_n_operation.extend(
                separation_info.get('vehicle_operation_pairs', [])
            )
    
    def _handle_resource_available(self, event: Event):
        """
        Handle resource becoming available after separation time.
        Now unified to handle both:
        1. Full resource availability (resource becomes IDLE)
        2. Partial availability updates (specific vehicles can now use resource)
        """
        if event.resource_id is None or event.resource_id not in self.resources:
            return
            
        resource = self.resources[event.resource_id]
        
        # Check if this is a partial separation update or full availability
        is_separation_update = (event.data and 
                               event.data.get('event_subtype') == 'separation_update')
        
        if is_separation_update:
            # Handle partial separation time update
            if event.data and 'vehicle_operation_pairs' in event.data:
                new_allocatable_veh_oper_pairs = event.data['vehicle_operation_pairs']
                for pair in new_allocatable_veh_oper_pairs:
                    if pair in resource.allocation_prohibited_vehicle_n_operation:
                        resource.allocation_prohibited_vehicle_n_operation.remove(pair)

            # Remove this event ID from pending list
            if event.event_id and event.event_id in resource.separation_time_reached_events:
                resource.separation_time_reached_events.remove(event.event_id)
            
            self.logger.debug("Updated allocatable vehicles for resource %d at time %.2f remaining separation time list : %s", 
                             resource.id, self.current_time, resource.allocation_prohibited_vehicle_n_operation)
            if not resource.separation_time_reached_events:
                if resource.state == ResourceState.SEPARATION_DELAY:
                    resource.state = ResourceState.IDLE
                    self.logger.debug("Resource %d became IDLE at time %.2f", 
                    resource.id, self.current_time)
                resource.allocation_prohibited_vehicle_n_operation.clear()
        else:
            # Handle full resource availability (traditional behavior)
            resource.state = ResourceState.IDLE
            resource.allocation_prohibited_vehicle_n_operation.clear() 
            self.logger.debug("Resource %d became IDLE at time %.2f", 
                             resource.id, self.current_time)
        
        # Note: Resource assignment is handled externally
    
    def _handle_planned_departure_reached(self, event: Event):
        """Handle planned departure reached event."""
        vehicle = self.vehicles[event.vehicle_id]
        self.logger.debug("Planned departure time reached for vehicle %d at time %.2f", vehicle.id, self.current_time)

    def _handle_vehicle_departure(self, event: Event):
        """Handle aircraft departure from vertiport"""
        vehicle = self.vehicles[event.vehicle_id]
        vehicle.state = VehicleState.DEPARTED
        
        # Calculate tardiness
        arrival_tardiness = max(0, vehicle.log_operation_finish_times[0] - vehicle.planned_arrival_time)
        departure_tardiness = max(0, self.current_time - vehicle.planned_departure_time)
        
        self.stats['tardiness']['arrival'].append(arrival_tardiness)
        self.stats['tardiness']['departure'].append(departure_tardiness)
        
        # Calculate flow time
        flow_time = self.current_time - vehicle.arrival_time
        self.stats['total_flow_time'] += flow_time
        
        # Calculate waiting time
        total_processing_time = sum(vehicle.log_operation_finish_times[i] - vehicle.log_start_times[i] 
                                  for i in range(len(vehicle.log_start_times)))
        waiting_time = flow_time - total_processing_time
        self.stats['total_waiting_time'] += waiting_time
        
        self.stats['completed_vehicles'] += 1
        
        self.logger.debug("Vehicle %d departed at time %.2f", vehicle.id, self.current_time)
    
    def _calculate_separation_time(self, resource: Resource, operation: int, vehicle: Vehicle, next_vehicle: Optional[Vehicle] = None) -> float:
        """
        Calculate separation time after an operation based on:
        1. Only pad operations (landing/takeoff) have separation requirements
        2. Separation time depends on operation pair and vehicle type combination
        3. Uses instance ST dictionary: ST[operation_pair][prev_vehicle_id][next_vehicle_id][resource_index]
        
        If next_vehicle is None, returns a conservative estimate based on worst-case scenario.
        """
        if resource.resource_type != 'pad':
            return 0.0  # Only pads have separation requirements
        
        # Get resource index (pad index)
        resource_idx = resource.id
        
        # If no next vehicle is provided, calculate conservative separation time
        if next_vehicle is None:
            return self._calculate_conservative_separation_time(resource, operation, vehicle, resource_idx)
        
        # Calculate exact separation time with known next vehicle
        return self._calculate_exact_separation_time(resource, operation, vehicle, next_vehicle, resource_idx)
    
    def _calculate_list_of_separation_time(self, resource: Resource, operation: int, 
                                         vehicle: Vehicle, candidate_vehicles: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Calculate separation times for all candidate vehicles and return when each becomes allocatable.
        
        Args:
            resource: The resource that just completed an operation
            operation: The operation that just completed
            vehicle: The vehicle that just completed the operation
            candidate_vehicles: List of candidate vehicle IDs. If None, uses all vehicles in waiting queues
            
        Returns:
            List of dictionaries with keys:
            - 'vehicle_id': ID of the candidate vehicle
            - 'operation': Operation the candidate vehicle will perform
            - 'separation_time': Required separation time for this vehicle
            - 'available_time': Absolute time when resource becomes available for this vehicle
            - 'vehicle_type': Type of the candidate vehicle
        """
        if resource.resource_type != 'pad':
            return []  # Only pads have separation requirements
        
        resource_idx = resource.id
        
        # Get operation completion time
        operation_complete_time = resource.log_operation_finish_times[-1]
        
        # Determine candidate vehicles if not provided
        if candidate_vehicles is None:
            candidate_vehicles = list(range(len(self.vehicles)))
        
        separation_info = []
        last_op_idx = self.instance.num_operations - 1
        
        for candidate_id in candidate_vehicles:
            if candidate_id not in self.vehicles:
                continue
                
            candidate_vehicle = self.vehicles[candidate_id]
            
            # Check every operation index (not just current operation)
            for candidate_operation in [0, last_op_idx]:  # Landing and takeoff operations
                # Create operation pair key
                op_pair = (operation, candidate_operation)
                
                if op_pair in self.instance.ST:
                    try:
                        ST_matrix = self.instance.ST[op_pair]
                        separation_time = ST_matrix[vehicle.id][candidate_id][resource_idx]
                    except (KeyError, IndexError, TypeError):
                        raise ValueError("error from separation time handling")

                # Calculate when resource becomes available for this candidate
                available_time = operation_complete_time + separation_time
                
                # Add to results with vehicle-operation combination format
                separation_info.append({
                    'vehicle_operation_pair': (candidate_id, candidate_operation),
                    'vehicle_id': candidate_id,
                    'operation': candidate_operation,
                    'separation_time': separation_time,
                    'available_time': available_time,
                    'vehicle_type': candidate_vehicle.vehicle_type,
                    'operation_pair': op_pair
                })
        
        # Group separation_info by separation time value
        grouped_separtion_time_info = defaultdict(list)
        
        for info in separation_info:
            separation_time = info['separation_time']
            grouped_separtion_time_info[separation_time].append(info)
        
        # Convert to list of groups, sorted by separation time
        list_of_separation_time = []
        for separation_time in sorted(grouped_separtion_time_info.keys()):
            separation_time_elem = {
                'separation_time': separation_time,
                'available_time': operation_complete_time + separation_time,
                'vehicle_operation_pairs': grouped_separtion_time_info[separation_time]
            }
            list_of_separation_time.append(separation_time_elem)
        
        return list_of_separation_time

    def _calculate_conservative_separation_time(self, resource: Resource, operation: int, 
                                              vehicle: Vehicle, resource_idx: int) -> float:
        """
        Calculate a conservative separation time when the next vehicle is unknown.
        Uses worst-case scenario from the separation time matrix.
        """
        # Ensure this is a pad resource
        if resource.resource_type != 'pad':
            return 0.0
            
        last_op_idx = self.instance.num_operations - 1
        
        # Determine possible next operations
        possible_next_ops = [0, last_op_idx]  # Landing and takeoff
        
        max_separation = 0.0
        
        for next_op in possible_next_ops:
            op_pair = (operation, next_op)
            
            if op_pair not in self.instance.ST:
                continue
            
            separation_matrix = self.instance.ST[op_pair]
            
            # Find worst-case separation time for this vehicle against any possible next vehicle
            if vehicle.id < len(separation_matrix):
                vehicle_row = separation_matrix[vehicle.id]
                for next_vehicle_separations in vehicle_row:
                    if resource_idx < len(next_vehicle_separations):
                        separation_time = next_vehicle_separations[resource_idx]
                        max_separation = max(max_separation, separation_time)
        
        return max(0.0, max_separation)
    
    def _calculate_exact_separation_time(self, resource: Resource, operation: int, 
                                       vehicle: Vehicle, next_vehicle: Vehicle, resource_idx: int) -> float:
        """
        Calculate exact separation time when both vehicles are known.
        """
        # Ensure this is a pad resource
        if resource.resource_type != 'pad':
            return 0.0
            
        # Determine the operation pair key
        current_op = operation
        next_op = next_vehicle.current_operation
        
        # Create operation pair key
        op_pair = (current_op, next_op)
        
        # Check if this operation pair exists in ST dictionary
        if op_pair not in self.instance.ST:
            return 0.0
        
        try:
            # Get separation time from ST dictionary
            separation_matrix = self.instance.ST[op_pair]
            if (vehicle.id < len(separation_matrix) and 
                next_vehicle.id < len(separation_matrix[vehicle.id]) and
                resource_idx < len(separation_matrix[vehicle.id][next_vehicle.id])):
                
                separation_time = separation_matrix[vehicle.id][next_vehicle.id][resource_idx]
                return max(0.0, separation_time)
            else:
                return 2.0  # Default if indices are out of bounds
        except (KeyError, IndexError, TypeError):
            return 2.0  # Default if error occurs

    def _get_operation_state(self, operation: int, in_progress: bool) -> VehicleState:
        """Get vehicle state for a given operation"""
        state_map = {
            0: VehicleState.LANDING if in_progress else VehicleState.WAITING_FOR_LANDING,
            1: VehicleState.IN_BUFFER_IN if in_progress else VehicleState.WAITING_AFTER_LANDING,
            -2: VehicleState.AT_GATE,  # Gate is usually second-to-last or middle
            -1: VehicleState.TAKING_OFF if in_progress else VehicleState.WAITING_FOR_TAKEOFF_AT_BUFFER
        }
        
        # Handle different operation configurations
        if self.instance.num_operations == 3:  # Landing, Gate, Takeoff
            if operation == 1:
                return VehicleState.AT_GATE
            elif operation == 2:
                return VehicleState.TAKING_OFF if in_progress else VehicleState.WAITING_FOR_TAKEOFF_AT_BUFFER
        elif self.instance.num_operations == 4:  # With one buffer
            if operation == 1:
                return VehicleState.IN_BUFFER_IN if in_progress else VehicleState.WAITING_AFTER_LANDING
            elif operation == 2:
                return VehicleState.AT_GATE
            elif operation == 3:
                return VehicleState.TAKING_OFF if in_progress else VehicleState.WAITING_FOR_TAKEOFF_AT_BUFFER
        elif self.instance.num_operations == 5:  # With both buffers
            if operation == 1:
                return VehicleState.IN_BUFFER_IN if in_progress else VehicleState.WAITING_AFTER_LANDING
            elif operation == 2:
                return VehicleState.AT_GATE
            elif operation == 3:
                return VehicleState.IN_BUFFER_OUT if in_progress else VehicleState.WAITING_AFTER_GATE
            elif operation == 4:
                return VehicleState.TAKING_OFF if in_progress else VehicleState.WAITING_FOR_TAKEOFF_AT_BUFFER
        
        return state_map.get(operation, VehicleState.WAITING_FOR_LANDING)
    
    def _build_resource_ranges(self) -> List[List[int]]:
        """Build resource ranges for each operation type"""
        num_pad = self.instance.num_pad
        num_buffer_in = self.instance.num_buffer_in
        num_gate = self.instance.num_gate
        num_buffer_out = self.instance.num_buffer_out
        is_unified_buffer = self.instance.is_unified_buffer

        if is_unified_buffer:
            if num_buffer_in == 0:
                return [[0, num_pad], [num_pad, num_pad + num_gate], [0, num_pad]]
            else:
                return [[0, num_pad], [num_pad, num_pad + num_buffer_in],
                       [num_pad + num_buffer_in, num_pad + num_buffer_in + num_gate],
                       [num_pad, num_pad + num_buffer_in], [0, num_pad]]
        else:
            if num_buffer_in == 0:
                if num_buffer_out == 0:
                    return [[0, num_pad], [num_pad, num_pad + num_gate], [0, num_pad]]
                else:
                    return [[0, num_pad], [num_pad, num_pad + num_gate],
                           [num_pad + num_gate, num_pad + num_gate + num_buffer_out], [0, num_pad]]
            else:
                if num_buffer_out == 0:
                    return [[0, num_pad], [num_pad, num_pad + num_buffer_in],
                           [num_pad + num_buffer_in, num_pad + num_buffer_in + num_gate], [0, num_pad]]
                else:
                    return [[0, num_pad], [num_pad, num_pad + num_buffer_in],
                           [num_pad + num_buffer_in, num_pad + num_buffer_in + num_gate],
                           [num_pad + num_buffer_in + num_gate, num_pad + num_buffer_in + num_gate + num_buffer_out],
                           [0, num_pad]]
    
    def _generate_solution(self) -> Solution:
        """Generate Solution object from simulation results"""
        num_vehicles = self.instance.num_vehicles
        num_operations = self.instance.num_operations
        weights = self.instance.objective_weights
        
        # Initialize arrays
        start_times = np.zeros((num_vehicles, num_operations))
        finish_times = np.zeros((num_vehicles, num_operations))
        assigned_resources = np.zeros((num_vehicles, num_operations))
        arrival_time_tardiness = np.zeros(num_vehicles)
        departure_time_tardiness = np.zeros(num_vehicles)

        # Fill arrays from vehicle data
        for vehicle in self.vehicles.values():
            v_id = vehicle.id
            for op in range(min(len(vehicle.log_start_times), num_operations)):
                start_times[v_id, op] = vehicle.log_start_times[op]
                finish_times[v_id, op] = vehicle.log_operation_finish_times[op]
                assigned_resources[v_id, op] = vehicle.log_assigned_resources[op]
        
        # Calculate tardiness (following Solver._extract_solution pattern)
        for i in range(num_vehicles):
            # Arrival tardiness: actual arrival completion time - planned arrival time
            actual_arrival_completion = finish_times[i, 0]  # first operation (landing) completion
            planned_arrival_time = self.instance.vehicle_planned_arrival_times[i]
            arrival_time_tardiness[i] = max(0, actual_arrival_completion - planned_arrival_time)
            
            # Departure tardiness: actual departure start time - planned departure time
            actual_departure_start = start_times[i, num_operations - 1]  # last operation (takeoff) start
            planned_departure_time = self.instance.vehicle_planned_departure_times[i]
            departure_time_tardiness[i] = max(0, actual_departure_start - planned_departure_time)

        # Calculate objective value
        obj_val = (weights[0] * arrival_time_tardiness.sum() + 
                  weights[1] * departure_time_tardiness.sum())
        
        # Calculate total simulation time
        runtime = self.current_time
        
        return Solution(
            obj_val=obj_val,
            runtime=runtime,
            start_times=start_times,
            finish_times=finish_times,
            assinged_resources=assigned_resources,
            arrival_time_tardiness=arrival_time_tardiness,
            departure_time_tardiness=departure_time_tardiness,
            resource_ind=self._build_resource_ranges(),
            solver_type="DiscreteEventSimulation",
            instance=self.instance
        )

