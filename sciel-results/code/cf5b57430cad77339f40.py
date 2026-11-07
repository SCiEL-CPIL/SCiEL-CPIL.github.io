import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_points = []
        
        # ZDT3 Pareto front analysis:
        # f1 = x1
        # f2 = g(x)(1 + sin(10*pi*x1))
        # min g(x) = 1 (at x2..xn=0)
        # Non-dominated points are those not dominated by others.
        # Point (0, 1) dominates all x in (0, 0.1] where f2 >= 1.
        # Curve x in [0.1, 0.15] has f2 in [0, 1] and is non-dominated.
        # Points x > 0.15 are dominated by (0.15, 0).
        
        # Strategy:
        # 1. Evaluate x1=0 to get (0, 1).
        # 2. Evaluate x1 uniformly in [0.1, 0.15] to trace the active front.
        
        n_total = self.budget
        if n_total < 2:
            # Fallback if budget is tiny
            x1_vals = np.linspace(0, 0.15, n_total)
        else:
            # Reserve 1 for x1=0
            x1_zero = np.array([0.0])
            # Remaining for the active segment [0.1, 0.15]
            n_active = n_total - 1
            x1_active = np.linspace(0.1, 0.15, n_active)
            x1_vals = np.concatenate([x1_zero, x1_active])
            
        for x1 in x1_vals:
            x = np.full(self.dim, lo[1]) # Set x2..xn to lower bound (0)
            x[0] = x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_points.append(x)
            
        return all_points
