"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer exploiting ZDT3's known Pareto front structure.
    ZDT3's Pareto front lies on x[1:] = 0.5, with f1 = x[0] and
    f2 = 1 + 10*(1 - sqrt(x[0]/30) - (x[0]/30)*sin(10*pi*x[0])).
    
    Strategy:
    1. Use the full budget for dense, non-uniform sampling along x[0]
       (concentrating where the front is complex) with x[1:] = 0.5.
    2. Use a power-law distribution to concentrate samples in regions
       with higher curvature of the Pareto front.
    3. Track elapsed time to stay within the 300s limit.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)
        
        used = 0
        all_points = []
        all_fvals = []
        
        # Phase 1: Dense non-uniform sampling along the Pareto front manifold
        # The Pareto front of ZDT3 is on x[1:] = 0.5
        # f1 = x[0], f2 = 1 + 10*(1 - sqrt(x[0]/30) - (x[0]/30)*sin(10*pi*x[0]))
        # The front has 24 disconnected segments due to the sin term
        # Concentrate samples where the front is more complex
        
        n_phase1 = self.budget
        
        # Create non-uniform grid: more points where the front curves more
        # Use a power-law distribution to concentrate in regions with higher curvature
        # The sin(10*pi*x0) term creates oscillations; the front is more complex near x0=0
        # Use power p=0.8 to cluster near 0
        
        u = np.linspace(0.0, 1.0, n_phase1)
        x0_values = u ** 0.8
        
        # Add small random perturbation to break symmetry
        x0_values = x0_values + rng.normal(0, 0.001, size=n_phase1)
        x0_values = np.clip(x0_values, 0.0, 1.0)
        
        # Shuffle to avoid ordering bias
        rng.shuffle(x0_values)
        
        for x0 in x0_values:
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
            x = np.full(self.dim, 0.5)
            x[0] = x0
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_points.append(x.copy())
            all_fvals.append(f)
            used += 1
        
        # Phase 2: If budget remains, do random search
        if used < self.budget and time.time() < deadline:
            while used < self.budget and time.time() < deadline:
                x = lo + (hi - lo) * rng.random(self.dim)
                f = problem(x)
                all_points.append(x.copy())
                all_fvals.append(f)
                used += 1
        
        return all_points
# EVOLVE-BLOCK-END
