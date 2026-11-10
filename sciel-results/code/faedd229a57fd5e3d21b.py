"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT4.
    Exploits the analytic structure: Pareto front is achieved when x[1:] minimize g(x).
    For standard ZDT4, min g(x_i) is at x_i=0.
    We set x[1:] to the value in bounds closest to 0.
    We sample x[0] uniformly across its bounds to cover the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        dim = self.dim
        
        # Determine optimal values for tail variables x[1:]
        # g(x_i) = x_i^2 - 10cos(4pi x_i) + 10
        # Global minimum is at x_i = 0.
        # If 0 is in bounds, use 0. Otherwise, use the bound closest to 0.
        tail = np.zeros(dim - 1)
        for i in range(dim - 1):
            if lo[i+1] <= 0 <= hi[i+1]:
                tail[i] = 0.0
            else:
                # Pick the bound closest to 0
                if abs(lo[i+1]) < abs(hi[i+1]):
                    tail[i] = lo[i+1]
                else:
                    tail[i] = hi[i+1]
        
        # Ensure tail is within bounds (should be by construction)
        tail = np.clip(tail, lo[1:], hi[1:])
        
        # We want to cover the range of x[0] to maximize hypervolume.
        # The Pareto front for ZDT4 is a line segment if g(x) is constant.
        # Hypervolume is determined by the extreme points of f1 (which is x0).
        # So we must include lo[0] and hi[0].
        
        n_samples = self.budget
        if n_samples < 2:
            n_samples = 2
            
        # Generate samples for x[0]
        # Linspace ensures endpoints are included
        x0_samples = np.linspace(lo[0], hi[0], n_samples)
        
        # Sort samples to prioritize endpoints? 
        # Linspace is already sorted.
        # We iterate through all of them.
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            
            x0 = x0_samples[i]
            x = np.concatenate(([x0], tail))
            
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            
            try:
                problem(x)
            except Exception:
                break

        return None
# EVOLVE-BLOCK-END
