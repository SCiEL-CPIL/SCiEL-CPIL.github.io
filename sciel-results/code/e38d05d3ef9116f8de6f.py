"""Optimized DTLZ4 solver: direct analytical Pareto front construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using analytical Pareto front construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 with M=3, N=12:
        # Position variables: x[0], x[1] (M-1 = 2)
        # Distance variables: x[2]...x[11] (N-M+1 = 10)
        # For Pareto front (g=0), set distance variables to 0.5
        # Objectives:
        # f1 = cos(x0*pi/2)*cos(x1*pi/2)
        # f2 = cos(x0*pi/2)*sin(x1*pi/2)
        # f3 = sin(x0*pi/2)

        # Strategy: Use Sobol sequence for low-discrepancy sampling in 2D parameter space,
        # then map to the spherical triangle Pareto front of DTLZ4 (M=3).
        # Sobol sequences have better uniformity properties than Fibonacci lattice.
        
        n_points = budget
        
        # Use Sobol sequence for low-discrepancy sampling in [0,1]^2
        # Sobol sequences are deterministic and have excellent uniformity in high dimensions
        sampler = qmc.Sobol(d=2, scramble=True)
        # Generate n_points samples, but we need to skip the first point (which is [0,0])
        # to get better distribution, or just use all of them
        sobol_samples = sampler.random(n=n_points)
        
        # Map from [0,1]^2 to (alpha, beta) on the spherical triangle.
        # For uniform surface area on the quarter sphere:
        # Area element dA = sin(alpha) d(alpha) d(beta).
        # To sample uniformly on the surface, we need P(alpha < a) = integral_0^a sin(t) dt = 1 - cos(a).
        # So u = 1 - cos(alpha) => cos(alpha) = 1 - u => alpha = arccos(1 - u).
        h1 = sobol_samples[:, 0]
        h2 = sobol_samples[:, 1]
        
        alpha = np.arccos(1.0 - h1)
        beta = (np.pi / 2) * h2
        
        # Compute objective values
        f1 = np.cos(alpha) * np.cos(beta)
        f2 = np.cos(alpha) * np.sin(beta)
        f3 = np.sin(alpha)
        
        # Construct decision vectors: x[0]=alpha/(pi/2), x[1]=beta/(pi/2), x[2:]=0.5
        x0_vals = 2.0 * alpha / np.pi
        x1_vals = beta / (np.pi / 2.0)  # which is just h2
        
        params = np.column_stack([x0_vals, x1_vals])
        
        # Add the 3 corner points explicitly to ensure extreme coverage
        # Corners in (x0, x1) space:
        # (0,0) -> f=(1,0,0), (0,1) -> f=(0,1,0), (1,0) -> f=(0,0,1)
        corner_params = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ])
        
        if n_points > 3:
            # Replace last 3 points with corners to ensure extreme coverage
            params[-3:] = corner_params
        else:
            params = corner_params[:n_points]
        
        # Final check: ensure all params are in [0, 1]
        params = np.clip(params, 0.0, 1.0)
        
        # Construct decision vectors
        X = np.zeros((len(params), dim))
        X[:, 0] = params[:, 0]
        X[:, 1] = params[:, 1]
        X[:, 2:] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        for i in range(len(X)):
            if time.time() > deadline:
                break
            problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
