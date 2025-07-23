from Generator import generate_instance
from Solver import solve
from Visualizer import visualize_result
from Instance import InstanceConfig

if __name__ == "__main__":
    print("This is a module for solving optimization problems using Gurobi.")
    config = InstanceConfig(num_vehicle=30, num_pad=3)  # Example: override defaults
    instance = generate_instance(config)
    solution = solve(instance)
    visualize_result(instance, solution)
