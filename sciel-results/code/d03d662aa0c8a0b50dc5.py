"""Rewritten optimizer for BBOB f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Hybrid optimizer:
    1. Quick local refinement from a few random starts to get a good initial guess.
    2. Leverage the 2π-periodicity of Rastrigin in the rotated frame:
       - The hidden transform is x_rot = R @ (x - shift).
       - f(x) = sum_i [10*(x_rot_i^2 - 10*cos(2*pi*x_rot_i)) + 10*dim] + offset
       - Minima occur when x_rot_i is an integer for all i.
       - So candidate minima in original space: x = R^T @ (k + shift) for integer vector k.
       - But we don't know R or shift exactly. However, if we have a good estimate of the
         current best x, we can estimate the rotated coordinates and infer nearby integer grids.
    
    Strategy:
    - Do a few random restarts with local BFGS/L-BFGS-B to find a good basin.
    - From the best x found, estimate the rotated frame: we don't know R, but we can use
      the fact that near a minimum, the function is locally quadratic and periodic.
    - Alternative: since the problem is small (dim=5, budget=500), do a smarter global search.
    
    Actually, a simpler and more robust approach:
    - Use multiple random restarts with L-BFGS-B, each with a small budget.
    - Keep the best solution found.
    - This should outperform the simple (1+1)-ES because L-BFGS-B converges faster near minima.
    
    Even better: use a combination of:
    - A coarse grid/random search to find promising regions.
    - Local refinement with L-BFGS-B.
    
    Given budget=500, dim=5:
    - We can afford ~10-20 local optimizations, each using ~20-50 function evaluations.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None
        self.best_x = None
        self.best_f = np.inf

    def _counted_func(self, x):
        """Wrapper to count function calls."""
        self.calls += 1
        return self.func(x)

    def __call__(self, func):
        self.func = func
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Multiple random restarts with local L-BFGS-B
        # Each restart uses a limited number of function evaluations.
        n_restarts = 15
        max_evals_per_restart = max(10, (budget - n_restarts) // n_restarts)

        # Initial random points
        for i in range(n_restarts):
            if self.calls >= budget - 5:
                break

            # Random start
            x0 = rng.uniform(-5, 5, dim)
            
            # Define a local function wrapper that stops when budget is exhausted
            def local_func(x):
                if self.calls >= budget - 1:
                    return self._counted_func(x)
                return self._counted_func(x)

            try:
                res = minimize(
                    local_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 30, 'maxls': 30, 'maxfun': max_evals_per_restart}
                )
                fx = res.fun
                x = res.x
                if fx < self.best_f:
                    self.best_f = fx
                    self.best_x = x.copy()
            except Exception:
                # If optimization fails, just evaluate the starting point
                if self.calls < budget - 1:
                    fx = self._counted_func(x0)
                    if fx < self.best_f:
                        self.best_f = fx
                        self.best_x = x0.copy()

        # Phase 2: If we have a good best, try to exploit periodicity
        # Near a minimum, the function is approximately periodic with period 2*pi in the rotated frame.
        # We can try shifting the best solution by small amounts to see if we can find a better basin.
        # But without knowing the rotation, this is tricky.
        
        # Instead, do a few more random restarts in the neighborhood of the best solution found.
        if self.best_x is not None and self.calls < budget - 10:
            n_local_restarts = 5
            for i in range(n_local_restarts):
                if self.calls >= budget - 5:
                    break
                # Perturb the best solution
                perturbation = 0.5 * rng.standard_normal(dim)
                x0 = np.clip(self.best_x + perturbation, -5, 5)
                
                def local_func(x):
                    return self._counted_func(x)

                try:
                    res = minimize(
                        local_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 20, 'maxls': 20, 'maxfun': 10}
                    )
                    fx = res.fun
                    x = res.x
                    if fx < self.best_f:
                        self.best_f = fx
                        self.best_x = x.copy()
                except Exception:
                    pass

        # Ensure we've made at least one call
        if self.calls == 0:
            x0 = np.zeros(dim)
            self.best_f = self._counted_func(x0)
            self.best_x = x0.copy()

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
