"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start optimizer: Sobol global sampling + Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Sobol sequence for quasi-random global sampling
        # Use scipy's Sobol sequence
        from scipy.stats import qmc

        # Phase 1: Global exploration with Sobol sequence
        n_global = min(40, budget)
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        points = sampler.random(n_global)  # [0, 1]^dim
        # Map to [-5, 5]^dim
        points = points * 10.0 - 5.0

        best_x = None
        best_f = np.inf
        all_results = []

        # Evaluate global points
        for i in range(n_global):
            x = points[i]
            fx = func(x)
            all_results.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Check time
            if time.time() > deadline:
                break

        # Sort by function value
        all_results.sort(key=lambda t: t[0])

        # Phase 2: Local refinement from top starting points
        n_local = budget - n_global
        if n_local > 0 and best_x is not None:
            # Take top k starting points for local refinement
            top_k = min(5, len(all_results))
            start_points = [all_results[i][1] for i in range(top_k)]

            calls_made = n_global
            for sp in start_points:
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Nelder-Mead local optimization
                try:
                    result = minimize(
                        func,
                        sp,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max(1, (budget - calls_made) // top_k),
                            'xatol': 1e-6,
                            'fatol': 1e-6,
                        }
                    )
                    calls_made += result.nfev
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
