"""Optimizer for BBOB f17 (Schaffers F7) using Sobol + Nelder-Mead refinement."""
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Sobol quasi-random global search interleaved with Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        # Sobol sequence for uniform coverage
        # scipy.stats.qmc.Sobol
        from scipy.stats.qmc import Sobol

        # We need enough Sobol points. Let's use 256 base points for global exploration
        n_sobol = 256
        sobol = Sobol(d=dim, scramble=True, seed=42)
        sobol_points = sobol.random(n_sobol)  # shape (n_sobol, dim), in [0,1]^dim
        # Map to [-5, 5]^dim
        sobol_points = sobol_points * 10.0 - 5.0

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Phase 1: Evaluate Sobol points to find a good global candidate
        # Interleave with occasional Nelder-Mead refinement on the best so far

        # We'll evaluate Sobol points in batches, and after each batch,
        # if we have enough budget, do a local refinement on the current best.

        # Strategy: evaluate all Sobol points first (256 calls), then use remaining
        # budget for local refinement via Nelder-Mead from the best point.
        # But to get early progress, let's do it in a smarter way:
        # Evaluate Sobol points one by one, and every ~32 points, do a quick NM refinement.

        # Actually, for AOCC, early progress matters a lot. Let's evaluate Sobol points
        # in order (they're quasi-random so early points are also well-distributed).
        # After evaluating all Sobol points, do NM refinement.

        # Let's be more aggressive: evaluate Sobol points, and when we find a new best,
        # we could do a quick local search. But NM is expensive. Let's just do all Sobol
        # first, then NM.

        # Evaluate Sobol points
        for i in range(n_sobol):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            x = sobol_points[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Nelder-Mead refinement from the best point found
        # Use remaining budget for NM. NM in 5D typically needs ~100-200 function evals.
        # We'll run NM with a max of remaining budget evaluations.

        if best_x is not None and calls_made < budget and time.time() < deadline:
            # Run Nelder-Mead from best_x
            # We need to count function evaluations during NM
            nm_calls = [0]
            nm_budget = budget - calls_made

            def nm_func(x):
                nonlocal nm_calls
                if nm_calls[0] >= nm_budget:
                    return best_f  # Return current best to stop early
                if time.time() > deadline:
                    return best_f
                nm_calls[0] += 1
                calls_made += 1
                return func(x)

            try:
                result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': nm_budget, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass

        # Phase 3: If we still have budget, do another round of local refinement
        # from a few of the best Sobol points, or try a second NM from a different start
        if calls_made < budget and time.time() < deadline and best_x is not None:
            # Try NM from the center of the box as a different starting point
            # This might find a different local minimum
            nm_calls2 = [0]
            nm_budget2 = min(100, budget - calls_made)

            def nm_func2(x):
                nonlocal nm_calls2
                if nm_calls2[0] >= nm_budget2:
                    return best_f
                if time.time() > deadline:
                    return best_f
                nm_calls2[0] += 1
                calls_made += 1
                return func(x)

            try:
                x0 = np.zeros(dim)  # Start from center
                result2 = minimize(
                    nm_func2,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': nm_budget2, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result2.fun < best_f:
                    best_f = result2.fun
                    best_x = result2.x
            except Exception:
                pass

        # Phase 4: Final polish with a small step-size local search if budget remains
        if calls_made < budget and time.time() < deadline and best_x is not None:
            # Simple coordinate descent / local random search for final polish
            rng = np.random.default_rng(7)
            sigma = 0.1
            for _ in range(min(100, budget - calls_made)):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.9

        return best_x, best_f
