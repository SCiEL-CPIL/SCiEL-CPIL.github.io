"""Optimizer for BBOB f13 (Sharp ridge) using Nelder-Mead with coordinate-descent init."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with coordinate-descent warm start and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # 300s limit, 10s safety margin
        calls = 0
        dim = self.dim

        def call(x):
            nonlocal calls
            calls += 1
            return func(x)

        # Phase 1: Coordinate descent to get a decent starting point
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, dim)
        fx = call(x)
        calls_used = calls

        # Coordinate descent: sweep each coordinate with a few steps
        step = 2.0
        for sweep in range(3):
            improved = False
            for d in range(dim):
                for direction in (1, -1):
                    for _ in range(20):
                        x_new = x.copy()
                        x_new[d] += direction * step
                        x_new = np.clip(x_new, -5, 5)
                        if np.array_equal(x_new, x):
                            break
                        if calls >= self.budget:
                            return x, fx
                        if time.time() > deadline:
                            return x, fx
                        fx_new = call(x_new)
                        if fx_new < fx:
                            x, fx = x_new, fx_new
                            improved = True
                        else:
                            break
            step *= 0.7
            if not improved:
                break

        if calls >= self.budget or time.time() > deadline:
            return x, fx

        # Phase 2: Nelder-Mead simplex method
        n = dim
        # Build initial simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x
        for i in range(n):
            simplex[i + 1] = x.copy()
            simplex[i + 1][i] += 0.5 if x[i] < 4.5 else -0.5
            simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)

        # Evaluate simplex
        fvals = np.zeros(n + 1)
        for i in range(n + 1):
            if calls >= self.budget or time.time() > deadline:
                return x, fx
            fvals[i] = call(simplex[i])

        # Track best
        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        best_f = fvals[best_idx]

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma = 0.5    # shrink

        no_improve_count = 0
        max_no_improve = 50

        while calls < self.budget and time.time() < deadline:
            if no_improve_count > max_no_improve:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            if calls >= self.budget or time.time() > deadline:
                break
            fr = call(xr)

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                if calls >= self.budget or time.time() > deadline:
                    break
                fe = call(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
                no_improve_count = 0
            elif fr < fvals[-2]:
                # Reflection accepted
                simplex[-1] = xr
                fvals[-1] = fr
                no_improve_count = 0
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                if calls >= self.budget or time.time() > deadline:
                    break
                fc = call(xc)

                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                    no_improve_count = 0
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls >= self.budget or time.time() > deadline:
                            break
                        fvals[i] = call(simplex[i])
                    no_improve_count += 1

            # Update best
            idx = np.argmin(fvals)
            if fvals[idx] < best_f:
                best_f = fvals[idx]
                best_x = simplex[idx].copy()

        # Final coordinate-descent polish if budget remains
        if calls < self.budget and time.time() < deadline:
            step = 0.1
            for sweep in range(5):
                improved = False
                for d in range(dim):
                    for direction in (1, -1):
                        for _ in range(30):
                            x_new = best_x.copy()
                            x_new[d] += direction * step
                            x_new = np.clip(x_new, -5, 5)
                            if np.array_equal(x_new, best_x):
                                break
                            if calls >= self.budget or time.time() > deadline:
                                return best_x, best_f
                            fx_new = call(x_new)
                            if fx_new < best_f:
                                best_x, best_f = x_new, fx_new
                                improved = True
                            else:
                                break
                step *= 0.6
                if not improved:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
