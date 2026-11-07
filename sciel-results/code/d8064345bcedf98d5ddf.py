"""Optimized Sphere (f1) minimizer with Nelder-Mead + adaptive restarts."""
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive restarts and time guard for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, budget_left, deadline, rng):
        """Run Nelder-Mead optimization with a budget cap and deadline."""
        n = self.dim
        # Build initial simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0
        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1, i] += 0.5

        # Evaluate initial simplex
        fvals = np.array([func(simplex[i]) for i in range(n + 1)])

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        calls = 0
        best_x = simplex[np.argmin(fvals)].copy()
        best_f = fvals.min()

        while calls < budget_left and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            # Check convergence
            if fvals[-1] - fvals[0] < 1e-10:
                break

            # Centroid of all but worst
            centroid = simplex[:-1].mean(axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            if time.time() >= deadline:
                break

            if fr < fvals[0]:
                # Try expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if time.time() >= deadline:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                if time.time() >= deadline:
                    break
                if fc <= min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if time.time() >= deadline:
                            break
                    if time.time() >= deadline:
                        break

            # Track best
            if fvals.min() < best_f:
                best_f = fvals.min()
                best_x = simplex[np.argmin(fvals)].copy()

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Phase 1: Use several random starting points with Nelder-Mead
        # Allocate budget across restarts
        n_restarts = 5
        per_restart_budget = max(10, budget // n_restarts)

        for r in range(n_restarts):
            if time.time() >= deadline:
                break
            remaining_budget = budget - calls_used
            if remaining_budget <= 0:
                break

            # Choose starting point
            if r == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            x_best, f_best, calls = self._nelder_mead(
                func, x0, min(per_restart_budget, remaining_budget), deadline, rng
            )
            calls_used += calls

            if f_best < best_f:
                best_f = f_best
                best_x = x_best

            # If we've converged very well, stop early
            if best_f < 1e-6:
                break

        # Phase 2: If we have budget left, do fine-tuning with smaller steps
        remaining = budget - calls_used
        if remaining > 5 and best_x is not None and time.time() < deadline:
            # Do a local (1+1)-ES fine-tuning
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(min(remaining, 50)):
                if time.time() >= deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85
            if fx < best_f:
                best_f = fx
                best_x = x

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
