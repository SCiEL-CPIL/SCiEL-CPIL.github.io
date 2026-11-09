"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer using Multi-Start Local Search with Random Initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Phase 1: Random Initialization
        n_init = min(50, self.budget)
        if n_init > 0:
            X_init = np.random.uniform(lo, hi, (n_init, self.dim))
            F_init = []
            for x in X_init:
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                self.eval_count += 1
                F_init.append(problem(x))
            
            F_init = np.array(F_init) if F_init else np.array([])
            
            # Find non-dominated points
            if len(F_init) > 0:
                # Simple non-dominated check
                # A point is non-dominated if no other point is better in all objectives
                # and better in at least one.
                # For small N, O(N^2) is fine.
                non_dominated_idx = []
                for i in range(len(F_init)):
                    is_nd = True
                    for j in range(len(F_init)):
                        if i == j: continue
                        # If j dominates i
                        if np.all(F_init[j] <= F_init[i]) and np.any(F_init[j] < F_init[i]):
                            is_nd = False
                            break
                    if is_nd:
                        non_dominated_idx.append(i)
                
                # Select up to 5 non-dominated points for refinement
                # If we have more, we might pick the ones with best sum or random
                # Let's just take the first 5
                candidates = [X_init[i] for i in non_dominated_idx[:5]]
            else:
                candidates = []
        else:
            candidates = []

        # If we didn't get 5 candidates, pad with random
        while len(candidates) < 5 and self.eval_count < self.budget:
            candidates.append(np.random.uniform(lo, hi))

        # Phase 2: Local Refinement
        # Weights for different regions of the Pareto front
        weights = [
            np.array([0.8, 0.1, 0.1]),
            np.array([0.1, 0.8, 0.1]),
            np.array([0.1, 0.1, 0.8]),
            np.array([0.4, 0.4, 0.2]),
            np.array([0.33, 0.33, 0.34])
        ]
        
        budget_per_refine = max(1, (self.budget - self.eval_count) // len(candidates))
        
        for i, x_start in enumerate(candidates):
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            
            w = weights[i % len(weights)]
            
            def obj(x, w=w):
                if self.eval_count >= self.budget or time.time() > deadline:
                    return np.inf
                self.eval_count += 1
                f = problem(x)
                return np.dot(w, f)
            
            try:
                # Use Nelder-Mead
                # maxfev should be the remaining budget for this run
                maxfev = min(budget_per_refine, self.budget - self.eval_count)
                if maxfev < 2:
                    continue
                res = minimize(obj, x_start, method='Nelder-Mead', 
                               options={'maxfev': maxfev, 'xatol': 1e-5, 'fatol': 1e-5})
            except Exception:
                pass

        return None
# EVOLVE-BLOCK-END
