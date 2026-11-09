"""Nelder-Mead based optimizer with multi-restart and time guard."""
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with deterministic initialization and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        # Nelder-Mead simplex method
        def nelder_mead(func, x0, f0, budget_remaining, dim):
            nonlocal best_x, best_f

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [f0]

            # Add vertices offset along axes
            step = 4.0
            for i in range(dim):
                v = x0.copy()
                v[i] += step
                v = np.clip(v, -5, 5)
                fv = func(v)
                simplex.append(v)
                fvals.append(fv)
                if fv < best_f:
                    best_f = fv
                    best_x = v.copy()
                budget_remaining -= 1
                if budget_remaining <= 0 or not time_ok():
                    return

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            # Sort by function value
            def sort_simplex():
                nonlocal simplex, fvals
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

            sort_simplex()

            while budget_remaining > 0 and time_ok():
                # Check convergence
                if max(fvals) - min(fvals) < 1e-12:
                    break

                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                budget_remaining -= 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fr < fvals[0]:
                    # Try expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    budget_remaining -= 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
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
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    budget_remaining -= 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            budget_remaining -= 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                            if budget_remaining <= 0 or not time_ok():
                                return

                sort_simplex()

            return

        # Multi-restart strategy with different initial points
        rng = np.random.default_rng(42)
        num_restarts = 8
        calls_per_restart = budget // num_restarts

        for restart in range(num_restarts):
            if not time_ok():
                break

            # Different starting points for each restart
            if restart == 0:
                x0 = np.zeros(dim)
            elif restart == 1:
                x0 = rng.uniform(-4, 4, dim)
            else:
                x0 = rng.uniform(-5, 5, dim)

            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            budget -= 1

            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            if budget <= 0 or not time_ok():
                break

            nelder_mead(func, x0, f0, calls_per_restart, dim)

            # If we used fewer calls than allocated, adjust
            if budget <= 0 or not time_ok():
                break

        return best_x, best_f
