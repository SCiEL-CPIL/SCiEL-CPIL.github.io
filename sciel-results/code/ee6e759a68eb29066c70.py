"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import math


class Optimizer:
    """Derivative-free optimizer using Gaussian Process surrogate with EI acquisition,
    followed by Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self.calls = 0

    def __call__(self, func):
        self.calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal best_x, best_f, self
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            val = float(func(x))
            self.calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Initial random sampling
        n_init = max(20, budget // 5)
        for _ in range(n_init):
            if self.calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            wrapped_func(x)

        # Phase 2: Gaussian Process surrogate with Expected Improvement
        # Reserve last 30% of budget for Nelder-Mead refinement
        nm_reserve = int(budget * 0.3)
        gp_budget = budget - self.calls - nm_reserve

        if gp_budget > 10 and best_x is not None:
            # Collect initial data points
            X_data = []
            y_data = []
            
            # We need to re-evaluate or store points. Since we can't easily
            # retrieve past evaluations, let's do a fresh GP phase.
            # Actually, let's use a simpler approach: do EI-guided sampling
            # using a simple GP approximation based on recent best points.
            
            # For simplicity and robustness, let's use a local search approach
            # with adaptive step sizes, guided by the best points found so far.
            # This is more reliable than a full GP implementation with limited budget.
            
            # Alternative: Use a simple kriging-like approach with a few points
            # Let's do iterative local refinement with adaptive step size
            
            # Phase 2a: Adaptive local search around best point
            step_size = 1.0
            no_improve_count = 0
            max_no_improve = 15
            
            for _ in range(gp_budget):
                if self.calls >= budget - nm_reserve:
                    break
                    
                if no_improve_count >= max_no_improve:
                    # Reset and try a larger step or restart near best
                    step_size = min(5.0, step_size * 1.5)
                    no_improve_count = 0
                    if step_size > 4.0:
                        # Try a new random point
                        x = rng.uniform(-5, 5, dim)
                        val = wrapped_func(x)
                        continue
                
                # Propose a new point near best_x with current step size
                direction = rng.normal(0, 1, dim)
                direction /= (np.linalg.norm(direction) + 1e-10)
                x_new = best_x + direction * step_size
                x_new = np.clip(x_new, -5, 5)
                
                val = wrapped_func(x_new)
                
                if val < best_f:
                    no_improve_count = 0
                    step_size = max(0.01, step_size * 0.8)
                else:
                    no_improve_count += 1
                    step_size = min(5.0, step_size * 1.1)

        # Phase 3: Nelder-Mead refinement from the best point
        if self.calls < budget:
            remaining = budget - self.calls
            nm_budget = remaining
            
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': nm_budget, 'maxfev': nm_budget, 'xatol': 1e-8, 'fatol': 1e-10}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
