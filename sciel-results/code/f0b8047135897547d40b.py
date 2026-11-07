"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """L-BFGS-B based optimizer with multi-start and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self._calls = 0
        self._max_calls = budget

    def __call__(self, func):
        self._calls = 0
        self._max_calls = self.budget
        rng = np.random.default_rng(42)

        # Wrapper to track calls and enforce budget
        def wrapped_func(x):
            if self._calls >= self._max_calls:
                return float('inf')
            self._calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            if not np.isfinite(val):
                return float('inf')
            return float(val)

        best_x = None
        best_f = float('inf')

        # Multi-start L-BFGS-B
        # We'll do several restarts with different initial points
        # Each L-BFGS-B call uses a limited number of iterations to stay within budget
        
        # Strategy: 
        # 1. Start with a few random points
        # 2. Use L-BFGS-B with bounded constraints
        # 3. If budget allows, do more restarts from best points + perturbation
        
        bounds = [(-5.0, 5.0)] * self.dim
        
        # First, try a few random initial points with L-BFGS-B
        # Tune maxiter to fit within budget: ~11 calls/iter for dim=5
        # 500 budget / 5 starts = 100 calls/start -> ~9 iters
        n_starts = 5
        for i in range(n_starts):
            if self._calls >= self._max_calls:
                break
            
            # Diversify initial points
            x0 = rng.uniform(-5, 5, self.dim)
            
            try:
                # Use L-BFGS-B with limited iterations
                # Reduced maxiter to allow multiple restarts within budget
                res = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 8, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass
        
        # If we still have budget, do more refined searches
        # Start from the best point found so far with small perturbations
        if best_x is not None and self._calls < self._max_calls:
            # Check remaining budget to decide refinement depth
            remaining_budget = self._max_calls - self._calls
            # Estimate cost per iteration ~11 calls
            max_iters_refine = max(1, int(remaining_budget / 11 / 3))
            
            n_refine = 3
            for i in range(n_refine):
                if self._calls >= self._max_calls:
                    break
                
                # Perturb the best point
                perturbation = 0.5 * rng.standard_normal(self.dim)
                x0 = np.clip(best_x + perturbation, -5.0, 5.0)
                
                try:
                    res = minimize(
                        wrapped_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': max(2, max_iters_refine), 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    
                    if res.fun < best_f and np.isfinite(res.fun):
                        best_f = res.fun
                        best_x = res.x.copy()
                        
                except Exception:
                    pass
        
        # If we still have budget, try even finer local searches
        if best_x is not None and self._calls < self._max_calls:
            remaining_budget = self._max_calls - self._calls
            max_iters_fine = max(1, int(remaining_budget / 11 / 2))
            
            n_fine = 2
            for i in range(n_fine):
                if self._calls >= self._max_calls:
                    break
                
                # Very small perturbation
                perturbation = 0.1 * rng.standard_normal(self.dim)
                x0 = np.clip(best_x + perturbation, -5.0, 5.0)
                
                try:
                    res = minimize(
                        wrapped_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': max(1, max_iters_fine), 'ftol': 1e-16, 'gtol': 1e-12}
                    )
                    
                    if res.fun < best_f and np.isfinite(res.fun):
                        best_f = res.fun
                        best_x = res.x.copy()
                        
                except Exception:
                    pass
        
        # Final evaluation to ensure we have a valid point
        if best_x is None:
            # Fallback: just use a random point
            best_x = rng.uniform(-5, 5, self.dim)
            if self._calls < self._max_calls:
                self._calls += 1
                best_f = wrapped_func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
