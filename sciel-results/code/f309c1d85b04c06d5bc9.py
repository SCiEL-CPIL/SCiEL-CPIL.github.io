"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with CMA-ES for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        calls = 0
        max_calls = self.budget

        best_x = np.zeros(self.dim)
        best_fx = float('inf')

        def counted_func(x):
            nonlocal calls, best_x, best_fx
            if calls >= max_calls:
                return 1e10
            if time.time() > deadline - 2.0:
                return 1e10
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse grid sampling to identify basin structure
        # Büche-Rastrigin has periodic ridges with period ~2*sqrt(2) in each dimension
        # Sample on a grid to find promising regions
        grid_n = 3  # 3x3x3x3x3 = 243 points
        grid_vals = np.linspace(-4.5, 4.5, grid_n)
        candidates = []
        
        for x0 in np.meshgrid(*[grid_vals] * self.dim, indexing='ij'):
            if calls >= max_calls - 100:
                break
            if time.time() > deadline - 10.0:
                break
            x_flat = np.array(x0).flatten()
            fx = counted_func(x_flat)
            candidates.append((fx, x_flat))
        
        # Sort by function value and take top candidates
        candidates.sort(key=lambda t: t[0])
        top_candidates = [c[1] for c in candidates[:min(20, len(candidates))]]
        
        # Phase 2: L-BFGS-B from multiple starting points
        lb = np.full(self.dim, -5.0)
        ub = np.full(self.dim, 5.0)
        
        # Add some random starts too
        n_random = 5
        for _ in range(n_random):
            if calls >= max_calls - 50:
                break
            x_rand = rng.uniform(-4, 4, self.dim)
            top_candidates.append(x_rand)
        
        # Run L-BFGS-B from each candidate
        for x0 in top_candidates:
            if calls >= max_calls - 30:
                break
            if time.time() > deadline - 5.0:
                break
            
            try:
                res = minimize(
                    counted_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=list(zip(lb, ub)),
                    options={'maxiter': 50, 'maxfun': 50}
                )
            except Exception:
                pass
        
        # Phase 3: Fine-tuning with small perturbations around best
        if calls < max_calls - 10 and time.time() < deadline - 3.0:
            sigma_fine = 0.05
            remaining = min(max_calls - calls, 50)
            for _ in range(remaining):
                if time.time() > deadline - 1.0:
                    break
                y = np.clip(best_x + sigma_fine * rng.standard_normal(self.dim), -5, 5)
                fy = counted_func(y)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
