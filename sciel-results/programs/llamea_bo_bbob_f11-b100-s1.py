"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart L-BFGS-B with Nelder-Mead polish for Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = None
        best_fx = np.inf
        calls = 0

        def safe_func(x):
            nonlocal calls, best_x, best_fx
            if calls >= budget:
                return best_fx
            x = np.clip(x, -5, 5)
            calls += 1
            fx = float(func(x))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-restart L-BFGS-B on a set of initial points
        # Use about 60% of budget for L-BFGS-B
        lbfgs_budget = int(budget * 0.60)

        # Create a set of diverse initial points
        # Mix of grid points and random points
        initial_points = []

        # Center point
        initial_points.append(np.zeros(d))

        # Points near corners of the search space
        for i in range(4):
            pt = np.zeros(d)
            for j in range(d):
                pt[j] = rng.choice([-3, 3])
            initial_points.append(pt)

        # Random points
        while len(initial_points) < 10:
            initial_points.append(rng.uniform(-4, 4, d))

        # Run L-BFGS-B from each initial point
        from scipy.optimize import minimize

        for i, x0 in enumerate(initial_points):
            if not time_ok() or calls >= lbfgs_budget:
                break

            # Check if we've already evaluated a point close to this one
            if best_x is not None and np.linalg.norm(x0 - best_x) < 0.1:
                continue

            def lbfgs_func(v):
                nonlocal calls
                if calls >= budget:
                    return best_fx
                return safe_func(v)

            try:
                res = minimize(
                    lbfgs_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * d,
                    options={'maxiter': 100, 'maxls': 50, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 2: Nelder-Mead polish from best point
        if time_ok() and best_x is not None:
            try:
                from scipy.optimize import minimize

                nm_calls = [0]

                def nm_func(v):
                    nonlocal calls
                    if nm_calls[0] >= budget - calls:
                        return best_fx
                    nm_calls[0] += 1
                    return func(np.clip(v, -5, 5))

                polish_budget = budget - calls
                if polish_budget > 10:
                    res = minimize(
                        nm_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': polish_budget, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                    )
                    calls += nm_calls[0]
                    if res.fun < best_fx:
                        best_x = np.clip(res.x, -5, 5)
                        best_fx = res.fun
            except Exception:
                pass

        # Phase 3: Additional L-BFGS-B restarts from points near the current best
        if time_ok() and best_x is not None and calls < budget - 50:
            try:
                from scipy.optimize import minimize

                # Generate perturbed points around the best
                for _ in range(5):
                    if not time_ok() or calls >= budget - 20:
                        break

                    perturbation = rng.normal(0, 0.5, d)
                    x0 = np.clip(best_x + perturbation, -5, 5)

                    def lbfgs_func2(v):
                        nonlocal calls
                        if calls >= budget:
                            return best_fx
                        return safe_func(v)

                    try:
                        res = minimize(
                            lbfgs_func2,
                            x0,
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * d,
                            options={'maxiter': 50, 'maxls': 30, 'ftol': 1e-15, 'gtol': 1e-12}
                        )
                        if res.fun < best_fx:
                            best_fx = res.fun
                            best_x = np.clip(res.x, -5, 5)
                    except Exception:
                        pass
            except Exception:
                pass

        # Final Nelder-Mead polish if we have budget left
        if time_ok() and best_x is not None and calls < budget - 10:
            try:
                from scipy.optimize import minimize

                nm_calls = [0]

                def nm_func2(v):
                    nonlocal calls
                    if nm_calls[0] >= budget - calls:
                        return best_fx
                    nm_calls[0] += 1
                    return func(np.clip(v, -5, 5))

                polish_budget = budget - calls
                if polish_budget > 10:
                    res = minimize(
                        nm_func2,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': polish_budget, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
                    )
                    calls += nm_calls[0]
                    if res.fun < best_fx:
                        best_x = np.clip(res.x, -5, 5)
                        best_fx = res.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
