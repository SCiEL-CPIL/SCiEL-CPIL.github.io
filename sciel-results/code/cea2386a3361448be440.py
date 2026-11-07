"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    """Hybrid Sobol Multi-Start with L-BFGS-B local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 250s out of 300s limit
        deadline = self.start_time + 250.0

        def wrapped_func(x):
            if self.calls >= self.budget:
                return 1e10
            self.calls += 1
            x_clipped = np.clip(x, -5, 5)
            try:
                val = func(x_clipped)
            except Exception:
                return 1e10
            if np.isnan(val) or np.isinf(val):
                return 1e10
            return val

        # 1. Global Phase: Sobol Sampling
        n_sobol = 20
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=0)
        points = sampler.random(n_sobol)
        points = points * 10.0 - 5.0

        evaluated_points = []
        for x in points:
            if self.calls >= self.budget or time.time() > deadline:
                break
            fx = wrapped_func(x)
            evaluated_points.append((fx, x))
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()

        # Sort by fitness to prioritize best starts for refinement
        evaluated_points.sort(key=lambda item: item[0])

        # 2. Local Phase: L-BFGS-B Refinement
        # L-BFGS-B with numerical gradients uses ~2*N+1 calls per iteration.
        # For N=5, that's 11 calls/iter.
        
        # Refine the top 3 starting points
        refinement_budgets = [10, 5, 5] # Max iterations per refinement
        
        for i in range(min(3, len(evaluated_points))):
            if self.calls >= self.budget - 20 or time.time() > deadline:
                break
            
            start_x = evaluated_points[i][1]
            max_iter = refinement_budgets[i]
            
            try:
                res = minimize(
                    wrapped_func, 
                    start_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except Exception:
                pass

        # 3. Additional Global Search if budget remains
        if self.calls < self.budget - 20 and time.time() < deadline:
            rng = np.random.default_rng(42)
            # Generate 10 new random points
            for _ in range(10):
                if self.calls >= self.budget - 20 or time.time() > deadline:
                    break
                x_new = rng.uniform(-5, 5, self.dim)
                fx_new = wrapped_func(x_new)
                if fx_new < self.best_f:
                    self.best_f = fx_new
                    self.best_x = x_new.copy()
                    # Polish this new best immediately
                    if self.calls < self.budget - 20:
                        try:
                            res = minimize(
                                wrapped_func, 
                                self.best_x, 
                                method='L-BFGS-B', 
                                bounds=[(-5, 5)] * self.dim,
                                options={'maxiter': 10, 'ftol': 1e-12, 'gtol': 1e-8}
                            )
                            if res.fun < self.best_f:
                                self.best_f = res.fun
                                self.best_x = res.x.copy()
                        except Exception:
                            pass

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
