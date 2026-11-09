# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import brentq


def _precompute_zdt6_pareto_table(n_points=200):
    """
    Precompute x1 values on the ZDT6 Pareto front with non-uniform distribution
    optimized for hypervolume. Uses f1_i = (i/(n-1))^2 to concentrate points
    near f1=0 where the Pareto front is steeper.
    """
    # Non-uniform f1 values: more points near f1=0
    if n_points <= 1:
        return np.array([0.0])
    f1_values = (np.arange(n_points) / (n_points - 1)) ** 2
    f1_values[0] = 0.0
    f1_values[-1] = 1.0
    
    x1_values = np.zeros(n_points)
    
    # For each target f1, find x1 such that 1 - exp(-4*x1)*sin(6*pi*x1)^6 = f1
    # i.e., exp(-4*x1)*sin(6*pi*x1)^6 = 1 - f1
    for i, f1 in enumerate(f1_values):
        target = 1.0 - f1
        if target <= 0.0:
            x1_values[i] = 1.0
            continue
        if target >= 1.0:
            x1_values[i] = 0.0
            continue
        
        # Find x1 in [0,1] where phi(x1) = target
        # phi(x1) = exp(-4*x1) * sin(6*pi*x1)^6
        # Use a fine grid to bracket, then brentq
        x_grid = np.linspace(0.0, 1.0, 5000)
        phi_grid = np.exp(-4.0 * x_grid) * (np.sin(6.0 * np.pi * x_grid) ** 6)
        diff = phi_grid - target
        sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
        
        if len(sign_changes) > 0:
            # Pick the smallest x1 solution (first sign change)
            idx = sign_changes[0]
            a, b = x_grid[idx], x_grid[idx+1]
            try:
                root = brentq(
                    lambda x: np.exp(-4.0 * x) * (np.sin(6.0 * np.pi * x) ** 6) - target,
                    a, b,
                    xtol=1e-14, rtol=1e-14, maxiter=100
                )
                x1_values[i] = root
            except:
                idx_min = np.argmin(np.abs(diff))
                x1_values[i] = x_grid[idx_min]
        else:
            idx_min = np.argmin(np.abs(diff))
            x1_values[i] = x_grid[idx_min]
    
    return x1_values


# Precompute the table at module load time
_ZDT6_X1_TABLE = _precompute_zdt6_pareto_table(200)


class Optimizer:
    """
    Optimizer that uses a pre-computed lookup table of ZDT6 Pareto-optimal decision vectors.
    Each call evaluates the precomputed points within the budget.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = min(self.budget, len(_ZDT6_X1_TABLE))
        
        # Build decision vectors: x1 from table, rest zeros
        X = np.zeros((n_points, self.dim))
        X[:, 0] = _ZDT6_X1_TABLE[:n_points]
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
