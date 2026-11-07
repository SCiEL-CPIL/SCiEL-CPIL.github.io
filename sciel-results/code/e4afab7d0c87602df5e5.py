"""New program: Hybrid Differential Evolution and Powell local optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls_made = 0
        self.budget_exceeded = False
        best_x = np.zeros(self.dim)
        best_f = float('inf')

        def counted_func(x):
            nonlocal best_x, best_f
            if self.budget_exceeded:
                return 0.0
            self.calls_made += 1
            if self.calls_made > self.budget:
                self.budget_exceeded = True
                raise Exception("Budget Exceeded")
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Global Search with Differential Evolution
        # Use a portion of the budget for global exploration.
        # DE is population-based, so it samples many points.
        # Dim=5. Popsize=5 -> 25 points per generation.
        # We'll allocate ~100 calls to DE (4 generations).
        
        if not self.budget_exceeded:
            try:
                bounds = [(-5, 5)] * self.dim
                # popsize=5, maxiter=4 -> ~100 calls
                # latinhypercube initialization provides good spread
                res_de = differential_evolution(
                    counted_func, 
                    bounds, 
                    popsize=5, 
                    maxiter=4, 
                    tol=1e-4, 
                    seed=42,
                    polish=False, 
                    init='latinhypercube'
                )
            except Exception:
                pass

        # Phase 2: Local Polish with Powell
        # Use the best point found so far.
        # Powell is very efficient for local convergence.
        if not self.budget_exceeded and self.calls_made < self.budget:
            try:
                # Use a higher maxiter for polish since we have budget left
                res_powell = minimize(
                    counted_func, 
                    best_x, 
                    method='Powell',
                    options={'maxiter': 100, 'xtol': 1e-8, 'ftol': 1e-8, 'disp': False}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
