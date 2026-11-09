"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead derivative-free optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0

        def count_func(x):
            nonlocal calls_used
            if calls_used >= budget:
                raise ValueError("Budget exceeded")
            calls_used += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        # Phase 1: Random initialization
        # Allocate some budget for random sampling to find diverse starting points
        n_random = 20
        xs = rng.uniform(-5, 5, (n_random, dim))
        fs = np.empty(n_random)
        for i in range(n_random):
            try:
                fs[i] = count_func(xs[i])
            except ValueError:
                break
        
        if calls_used >= budget:
            # Find best from what we have
            if n_random > 0:
                idx = np.argmin(fs[:calls_used])
                best_x = xs[idx].copy()
                best_f = fs[idx]
            return best_x, best_f

        # Find the best random point
        idx = np.argmin(fs)
        best_x = xs[idx].copy()
        best_f = fs[idx]

        # Sort the other points by fitness to use as secondary starts
        order = np.argsort(fs)
        starts = [xs[i].copy() for i in order if i != idx]

        # Phase 2: Nelder-Mead from best start
        # Nelder-Mead parameters:
        # We use adaptive=True to help it adjust to the landscape.
        
        def run_nm(start_x, max_iter):
            nonlocal best_x, best_f, calls_used
            try:
                res = minimize(
                    count_func,
                    start_x,
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except ValueError:
                # Budget exceeded, stop
                pass
            except Exception:
                pass

        # Estimate how many iterations we can afford.
        # Nelder-Mead cost per iteration is roughly dim+1 to 2*(dim+1).
        # Let's assume ~10 calls per iteration for dim=5.
        remaining = budget - calls_used
        max_iter_1 = max(10, remaining // 10)
        
        run_nm(best_x, max_iter_1)

        # Phase 3: If budget remains, try next best starts
        # Add a small perturbation to avoid identical basins
        if calls_used < budget - 10:
            for start in starts:
                if calls_used >= budget - 10:
                    break
                # Perturb start slightly
                start_perturbed = start + rng.normal(0, 0.1, dim)
                start_perturbed = np.clip(start_perturbed, -5, 5)
                
                remaining = budget - calls_used
                max_iter_2 = max(5, remaining // 15)
                run_nm(start_perturbed, max_iter_2)
                
        return best_x, best_f
# EVOLVE-BLOCK-END
