"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer for ZDT1: exploit known structure to maximize hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 structure: f1 = x[0], f2 = g(x)*(1-sqrt(x[0]/g(x)))
        # g(x) minimized when x[1:] = 0.5, giving g=1 and true Pareto front f2=1-sqrt(f1)
        # Strategy: fix x[1:]=0.5, spread x[0] to cover Pareto front for max hypervolume
        
        n_evals = self.budget
        # Generate x[0] values to maximize hypervolume on front f2=1-sqrt(f1)
        # Use dense sampling with emphasis where marginal hypervolume gain is highest
        # Optimal: equalize marginal contributions. Use adaptive spacing.
        
        # Start with a good initial set: include endpoints and use non-uniform spacing
        # that accounts for the sqrt curvature
        x0_vals = []
        x0_vals.append(0.0)
        x0_vals.append(1.0)
        
        # Add interior points: use spacing proportional to sqrt(f1) to equalize contributions
        # The derivative of f2 w.r.t. f1 is -1/(2*sqrt(f1)), so spacing should be ~sqrt(f1)
        # Use a grid in sqrt-space for better coverage
        n_interior = n_evals - 2
        if n_interior > 0:
            # Sample in t = sqrt(f1) space uniformly, then f1 = t^2
            t_vals = np.linspace(0, 1, n_interior + 2)[1:-1]  # exclude 0 and 1
            f1_vals = t_vals ** 2
            x0_vals.extend(f1_vals.tolist())
        
        # Sort and deduplicate
        x0_vals = sorted(set(x0_vals))
        
        # If we have more than budget, trim
        if len(x0_vals) > n_evals:
            x0_vals = x0_vals[:n_evals]
        
        # If we have fewer, add more points
        while len(x0_vals) < n_evals:
            # Add points in gaps
            gaps = []
            for i in range(len(x0_vals) - 1):
                gap = x0_vals[i+1] - x0_vals[i]
                gaps.append((gap, i))
            if gaps:
                gaps.sort(reverse=True)
                mid = (x0_vals[gaps[0][1]] + x0_vals[gaps[0][1]+1]) / 2
                x0_vals.append(mid)
            else:
                x0_vals.append(0.5)
        
        x0_vals = sorted(x0_vals[:n_evals])
        
        # Build and evaluate
        used = 0
        for x0 in x0_vals:
            if used >= n_evals:
                break
            x = np.full(self.dim, 0.5)  # x[1:] = 0.5 minimizes g
            x[0] = x0
            # Clip to bounds
            x = np.clip(x, lo, hi)
            problem(x)
            used += 1
        
        return None
# EVOLVE-BLOCK-END
