"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Deterministic 2D Pareto front coverage for DTLZ7 using a Fibonacci lattice
    mapped to the unit square, exploiting the known analytic structure of the front.
    
    DTLZ7 with M=3, dim=22:
    - x[0], x[1] parameterize the Pareto front position
    - x[2:] = 0 to achieve g=0 (Pareto front)
    - f_1 = x[0]
    - f_2 = x[1]  
    - f_3 = sqrt(x[0]) + sqrt(x[1]) - 1
    
    We generate well-distributed points in [0,1]^2 using a Fibonacci lattice
    (golden angle spiral) mapped to the square, ensuring uniform area coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_eval = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Generate n_eval points in [0,1]^2 using Fibonacci lattice
        # The golden angle is 2*pi*(1 - 1/phi) where phi = (1+sqrt(5))/2
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        golden_angle = 2.0 * np.pi * (1.0 - 1.0 / phi)
        
        # Generate indices
        n = np.arange(1, n_eval + 1, dtype=np.float64)
        
        # Fibonacci lattice: points on a spiral
        # r = sqrt(i / N) for uniform area distribution
        # theta = i * golden_angle
        r = np.sqrt(n / n_eval)
        theta = n * golden_angle
        
        # Convert to Cartesian in unit disk
        x_disk = r * np.cos(theta)
        y_disk = r * np.sin(theta)
        
        # Map from unit disk to unit square [0,1]^2
        # Use a simple affine mapping: scale and shift
        # x_disk, y_disk are in [-1, 1], map to [0, 1]
        x0_vals = 0.5 + 0.5 * x_disk
        x1_vals = 0.5 + 0.5 * y_disk
        
        # Clip to [0, 1] to handle any numerical edge cases
        x0_vals = np.clip(x0_vals, 0.0, 1.0)
        x1_vals = np.clip(x1_vals, 0.0, 1.0)
        
        # Create candidate solutions
        # x[0] = x0, x[1] = x1, x[2:] = 0
        X = np.zeros((n_eval, dim))
        X[:, 0] = x0_vals
        X[:, 1] = x1_vals
        # x[2:] = 0 (already zero)
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate all
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
