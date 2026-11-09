"""Nelder-Mead simplex with multiple restarts for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Nelder-Mead simplex with multiple restarts for ill-conditioned landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def call_func(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.where(np.isfinite(x), x, 0.0)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def nelder_mead(x0, max_calls):
            """Run Nelder-Mead simplex from x0, using at most max_calls evaluations."""
            nonlocal calls, best_x, best_f

            # Initialize simplex
            n = dim
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1, i] += 0.5  # Initial step size

            # Evaluate initial simplex
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                if calls >= budget or time.time() > deadline:
                    break
                fvals[i] = call_func(simplex[i])

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            # Track the best in the simplex
            best_simplex_f = np.min(fvals)
            best_simplex_idx = np.argmin(fvals)

            no_improve_count = 0
            max_no_improve = 20

            while calls < budget and (budget - calls) > 0:
                if time.time() > deadline:
                    break

                # Sort simplex by function values
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Check for convergence
                if np.max(fvals) - np.min(fvals) < 1e-12 * (1 + abs(np.min(fvals))):
                    break

                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                worst_point = simplex[-1]
                x_r = centroid + alpha * (centroid - worst_point)
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = call_func(x_r)

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst_point)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = call_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                    no_improve_count = 0
                elif f_r < fvals[-2]:
                    # Simple reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                    no_improve_count = 0
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        f_c = call_func(x_c)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - worst_point)
                        f_c = call_func(x_c)

                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = call_func(x_c) if np.isnan(f_c) else f_c

                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                        no_improve_count = 0
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = call_func(simplex[i])
                        no_improve_count += 1

                # Update best
                if np.min(fvals) < best_simplex_f - 1e-10:
                    best_simplex_f = np.min(fvals)
                    best_simplex_idx = np.argmin(fvals)
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count > max_no_improve:
                    break

            # Return the best point found
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # Multi-restart strategy
        rng = np.random.default_rng(42)

        # Initial random samples to find good starting points
        n_initial = min(30, budget)
        for _ in range(n_initial):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, dim)
            call_func(x)

        # Sort initial points and use best ones as starting points for Nelder-Mead
        # We'll do multiple Nelder-Mead runs from different starting points

        # Strategy: Do several Nelder-Mead runs, each with a portion of the budget
        # Start from the best few points found in initial sampling

        # Get top starting points (we need to track them)
        # Since we didn't track all points, let's do a few more targeted runs

        # First, run Nelder-Mead from the current best
        if best_x is not None:
            remaining = budget - calls
            if remaining > 10:
                x_opt, f_opt = nelder_mead(best_x.copy(), min(remaining, 200))

        # Then do additional restarts from random points
        n_restarts = 3
        for r in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break

            # Generate a new random starting point
            x_start = rng.uniform(-4, 4, dim)
            f_start = call_func(x_start)

            # Only proceed if this is competitive
            if f_start < best_f + 1.0:  # Within some range of best
                remaining = budget - calls
                if remaining > 10:
                    x_opt, f_opt = nelder_mead(x_start, min(remaining, 100))

        return best_x, best_f
# EVOLVE-BLOCK-END
