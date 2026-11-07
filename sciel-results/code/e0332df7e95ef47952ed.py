"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear-aware optimizer: detect gradient signs, jump to optimal vertex, verify."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Phase 1: Evaluate at origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls = 1
        x_best = x0.copy()
        fx_best = f0

        # Phase 2: Determine gradient sign for each coordinate
        # For linear f(x) = c + g·x, sign of g_i determines optimal x_i
        # If f(h*e_i) < f(0), then g_i < 0, so optimal x_i = +5
        # If f(h*e_i) > f(0), then g_i > 0, so optimal x_i = -5
        # If f(h*e_i) == f(0), then g_i ≈ 0, either extreme works
        h = 4.5  # Large step to clearly detect sign
        grad_signs = np.zeros(dim)

        for i in range(dim):
            if not time_ok() or calls >= budget:
                break
            e = np.zeros(dim)
            e[i] = h
            xp = np.clip(x0 + e, -5, 5)
            fp = func(xp)
            calls += 1
            if fp < f0:
                grad_signs[i] = -1.0  # g_i < 0, decrease in direction i, go to +5
            else:
                grad_signs[i] = 1.0   # g_i >= 0, increase in direction i, go to -5

        # Phase 3: Jump to predicted optimal vertex
        x_vertex = np.where(grad_signs > 0, -5.0, 5.0)
        fx_vertex = func(x_vertex)
        calls += 1
        if fx_vertex < fx_best:
            x_best = x_vertex.copy()
            fx_best = fx_vertex

        # Phase 4: Verify by checking the opposite vertex for a few coordinates
        # This helps if there's any non-linearity or if our sign detection was ambiguous
        # Try flipping one coordinate at a time from the vertex
        for i in range(dim):
            if not time_ok() or calls >= budget:
                break
            x_test = x_best.copy()
            # Flip coordinate i to the other extreme
            if x_best[i] > 0:
                x_test[i] = -5.0
            else:
                x_test[i] = 5.0
            fx_test = func(x_test)
            calls += 1
            if fx_test < fx_best:
                x_best = x_test.copy()
                fx_best = fx_test

        # Phase 5: If we have budget remaining, do a quick local search around best
        # Only if we haven't converged to a vertex (in case of non-linearity)
        if calls < budget and time_ok():
            # Check if we're at a vertex
            at_vertex = all(abs(abs(x_best[i]) - 5.0) < 1e-6 for i in range(dim))
            if at_vertex:
                # We're at a vertex; for a linear function this is optimal
                # Do minimal verification with a few random perturbations
                sigma = 0.5
                for _ in range(min(50, budget - calls)):
                    if not time_ok() or calls >= budget:
                        break
                    y = np.clip(x_best + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls += 1
                    if fy < fx_best:
                        x_best = y.copy()
                        fx_best = fy
                        sigma = min(sigma * 1.5, 2.0)
                    else:
                        sigma = max(sigma * 0.7, 0.01)
            else:
                # Not at vertex, do more thorough local search
                x = x_best.copy()
                fx = fx_best
                sigma = 1.0
                while calls < budget and time_ok():
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls += 1
                    if fy < fx:
                        x, fx = y.copy(), fy
                        if fx < fx_best:
                            x_best, fx_best = x.copy(), fx
                        sigma = min(sigma * 1.1, 3.0)
                    else:
                        sigma = max(sigma * 0.8, 1e-6)
                    if sigma < 1e-7:
                        break

        return x_best, fx_best
# EVOLVE-BLOCK-END
