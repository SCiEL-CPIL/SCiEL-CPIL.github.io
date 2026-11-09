"""Optimizer for BBOB f17 (Schaffer's F7) using Sobol quasi-random global search with adaptive local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Sobol quasi-random global search with adaptive Nelder-Mead refinement for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, leave 15s margin

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            fx = float(func(x))
            calls_made += 1
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        # Phase 1: Sobol quasi-random sampling for uniform coverage
        # Use Sobol sequence to efficiently cover the [-5, 5]^d space
        n_sobol = min(256, budget // 3)  # Use about 1/3 of budget for global exploration
        sobol = qmc.Sobol(d=d, scramble=True, seed=42)
        sobol_points = sobol.random(n_sobol)
        # Map from [0,1]^d to [-5,5]^d
        sobol_points = sobol_points * 10.0 - 5.0

        # Evaluate Sobol points
        for x in sobol_points:
            if not time_ok():
                break
            x = np.clip(x, -5, 5)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)

        # Phase 2: Adaptive local refinement using Nelder-Mead
        # Run multiple local optimizations from the best points found
        n_local = min(5, (budget - calls_made) // 20)
        
        # Get top diverse points from Sobol sampling
        # Re-evaluate to get all function values (we didn't store them, so use best and perturbations)
        candidate_starts = [best_x.copy()]
        
        # Add diverse starting points based on the Sobol points near the best
        # Generate additional diverse starting points
        for i in range(min(4, n_local - 1)):
            if not time_ok():
                break
            # Create perturbed version of best point
            scale = 1.5 / (i + 1)
            x_candidate = best_x + scale * rng.standard_normal(d)
            x_candidate = np.clip(x_candidate, -5, 5)
            candidate_starts.append(x_candidate)
        
        # Also try some of the Sobol points that were close to the best
        # Since we didn't store all values, let's just use random perturbations
        
        for x0 in candidate_starts:
            if not time_ok() or calls_made >= budget:
                break
            
            # Nelder-Mead local optimization
            try:
                result = minimize(
                    func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                # Count the calls made by this optimization
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 3: Further refinement with smaller step sizes
        if time_ok() and calls_made < budget:
            # Try Powell's method for further refinement
            try:
                result = minimize(
                    func,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 50,
                        'xatol': 1e-12,
                        'fatol': 1e-12
                    }
                )
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 4: Final local search with very small perturbations
        if time_ok() and calls_made < budget:
            n_final = min(20, budget - calls_made)
            for i in range(n_final):
                if not time_ok():
                    break
                scale = 0.1 / (i + 1)
                x_candidate = best_x + scale * rng.standard_normal(d)
                x_candidate = np.clip(x_candidate, -5, 5)
                fx = call_func(x_candidate)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x_candidate.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
