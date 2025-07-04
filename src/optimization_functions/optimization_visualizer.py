#!/usr/bin/env python3
"""
Optimization Functions Visualizer

A comprehensive toolkit for visualizing and analyzing optimization problems,
including 2D function visualization and constrained optimization.

Author: Extracted from ESE5460 HW0
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from typing import Callable, Tuple, List, Dict, Any


class OptimizationVisualizer:
    """A class for visualizing optimization problems and solutions."""
    
    def __init__(self):
        self.figsize = (10, 6)
        
    def six_hump_camel(self, x1: float, x2: float) -> float:
        """
        Six-hump camel function - a classic optimization test function.
        
        Args:
            x1, x2: Input coordinates
            
        Returns:
            Function value at (x1, x2)
        """
        return 2 * x1**2 - 1.05 * x1**4 + (1/6) * x1**6 - x1 * x2 + x2**2
    
    def plot_function_contour(self, 
                            func: Callable[[float, float], float],
                            x_range: Tuple[float, float] = (-3, 3),
                            y_range: Tuple[float, float] = (-3, 3),
                            resolution: int = 100,
                            num_contours: int = 300,
                            title: str = "Function Contour Plot") -> None:
        """
        Plot contour visualization of a 2D function.
        
        Args:
            func: Function to plot
            x_range: Range for x-axis
            y_range: Range for y-axis
            resolution: Grid resolution
            num_contours: Number of contour lines
            title: Plot title
        """
        x1 = np.linspace(x_range[0], x_range[1], resolution)
        x2 = np.linspace(y_range[0], y_range[1], resolution)
        X1, X2 = np.meshgrid(x1, x2)
        
        Y = func(X1, X2)
        
        plt.figure(figsize=self.figsize)
        contour = plt.contour(X1, X2, Y, num_contours)
        plt.colorbar(contour)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_3d_surface(self,
                       func: Callable[[float, float], float],
                       x_range: Tuple[float, float] = (-3, 3),
                       y_range: Tuple[float, float] = (-3, 3),
                       resolution: int = 100,
                       title: str = "3D Function Surface") -> None:
        """
        Plot 3D surface visualization of a 2D function.
        
        Args:
            func: Function to plot
            x_range: Range for x-axis
            y_range: Range for y-axis
            resolution: Grid resolution
            title: Plot title
        """
        x1 = np.linspace(x_range[0], x_range[1], resolution)
        x2 = np.linspace(y_range[0], y_range[1], resolution)
        X1, X2 = np.meshgrid(x1, x2)
        
        Y = func(X1, X2)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.8)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f(x1, x2)')
        ax.set_title(title)
        plt.colorbar(surface)
        plt.show()


class ConstrainedOptimizer:
    """A class for solving constrained optimization problems."""
    
    def __init__(self):
        self.optimization_results = []
    
    def quadratic_objective(self, x: np.ndarray) -> float:
        """
        Quadratic objective function: x1² + x2² - 6*x1*x2 - 4*x1 - 5*x2
        
        Args:
            x: Input vector [x1, x2]
            
        Returns:
            Function value
        """
        return x[0]**2 + x[1]**2 - 6*x[0]*x[1] - 4*x[0] - 5*x[1]
    
    def constraint1(self, x: np.ndarray, c: float = 4.0) -> float:
        """
        Inequality constraint: -(x1-2)² + c - x2 >= 0
        
        Args:
            x: Input vector [x1, x2]
            c: Constraint parameter
            
        Returns:
            Constraint value (positive means feasible)
        """
        return -(x[0] - 2)**2 + c - x[1]
    
    def constraint2(self, x: np.ndarray) -> float:
        """
        Inequality constraint: x1 + x2 - 1 >= 0
        
        Args:
            x: Input vector [x1, x2]
            
        Returns:
            Constraint value (positive means feasible)
        """
        return x[0] + x[1] - 1
    
    def solve_constrained_problem(self, 
                                initial_guess: Tuple[float, float] = (2, 0),
                                constraint_param: float = 4.0,
                                method: str = 'SLSQP') -> Dict[str, Any]:
        """
        Solve the constrained optimization problem.
        
        Args:
            initial_guess: Starting point for optimization
            constraint_param: Parameter for first constraint
            method: Optimization method
            
        Returns:
            Dictionary containing optimization results
        """
        constraints = [
            {'type': 'ineq', 'fun': lambda x: self.constraint1(x, constraint_param)},
            {'type': 'ineq', 'fun': lambda x: self.constraint2(x)}
        ]
        
        result = minimize(
            self.quadratic_objective,
            initial_guess,
            method=method,
            constraints=constraints
        )
        
        optimization_result = {
            'success': result.success,
            'optimal_point': result.x,
            'optimal_value': result.fun,
            'iterations': result.nit,
            'constraint_param': constraint_param,
            'message': result.message
        }
        
        self.optimization_results.append(optimization_result)
        return optimization_result
    
    def parameter_sensitivity_analysis(self, 
                                     param_range: Tuple[float, float] = (4.0, 4.2),
                                     num_points: int = 10) -> List[Dict[str, Any]]:
        """
        Analyze how the solution changes with constraint parameter.
        
        Args:
            param_range: Range of parameter values to test
            num_points: Number of parameter values to test
            
        Returns:
            List of optimization results for different parameters
        """
        param_values = np.linspace(param_range[0], param_range[1], num_points)
        results = []
        
        for param in param_values:
            result = self.solve_constrained_problem(constraint_param=param)
            results.append(result)
            
        return results
    
    def plot_sensitivity_analysis(self, results: List[Dict[str, Any]]) -> None:
        """
        Plot the results of parameter sensitivity analysis.
        
        Args:
            results: List of optimization results
        """
        params = [r['constraint_param'] for r in results]
        optimal_values = [r['optimal_value'] for r in results]
        x1_values = [r['optimal_point'][0] for r in results]
        x2_values = [r['optimal_point'][1] for r in results]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(params, optimal_values, 'o-')
        axes[0].set_xlabel('Constraint Parameter')
        axes[0].set_ylabel('Optimal Function Value')
        axes[0].set_title('Optimal Value vs Parameter')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(params, x1_values, 'o-', label='x1')
        axes[1].plot(params, x2_values, 's-', label='x2')
        axes[1].set_xlabel('Constraint Parameter')
        axes[1].set_ylabel('Optimal Point Coordinates')
        axes[1].set_title('Optimal Point vs Parameter')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(x1_values, x2_values, 'o-')
        axes[2].set_xlabel('x1')
        axes[2].set_ylabel('x2')
        axes[2].set_title('Optimal Point Trajectory')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """Demonstrate the optimization visualization and solving capabilities."""
    print("=== Optimization Functions Visualizer ===\n")
    
    # Create visualizer instance
    visualizer = OptimizationVisualizer()
    
    # 1. Visualize the six-hump camel function
    print("1. Visualizing Six-Hump Camel Function...")
    visualizer.plot_function_contour(
        visualizer.six_hump_camel,
        title="Six-Hump Camel Function - Contour Plot"
    )
    
    visualizer.plot_3d_surface(
        visualizer.six_hump_camel,
        title="Six-Hump Camel Function - 3D Surface"
    )
    
    # 2. Solve constrained optimization problem
    print("\n2. Solving Constrained Optimization Problem...")
    optimizer = ConstrainedOptimizer()
    
    # Solve with default parameter
    result1 = optimizer.solve_constrained_problem(constraint_param=4.0)
    print(f"Result with c=4.0: {result1}")
    
    # Solve with modified parameter
    result2 = optimizer.solve_constrained_problem(constraint_param=4.1)
    print(f"Result with c=4.1: {result2}")
    
    # 3. Parameter sensitivity analysis
    print("\n3. Parameter Sensitivity Analysis...")
    sensitivity_results = optimizer.parameter_sensitivity_analysis(
        param_range=(4.0, 4.2),
        num_points=10
    )
    
    optimizer.plot_sensitivity_analysis(sensitivity_results)
    
    print("\nAnalysis complete! Check the generated plots for insights.")


if __name__ == "__main__":
    main()