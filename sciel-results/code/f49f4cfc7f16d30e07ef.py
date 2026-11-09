"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        best_x = None
        best_f = float('inf')
        calls = 0

        def nelder_mead(x0, f0, max_calls):
            nonlocal calls, best_x, best_f
            dim = self.dim
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

            # Build initial simplex
            simplex = [x0.copy()]
            fvals = [f0]
            calls += 1

            for i in range(dim):
                xi = x0.copy()
                step = 0.5 if x0[i] > 0 else -0.5
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                simplex.append(xi)
                fvals.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            while calls < max_calls:
                if time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                xw = simplex[-1]
                fw = fvals[-1]
                xg = simplex[0]
                fg = fvals[0]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = np.clip(centroid + alpha * (centroid - xw), -5, 5)
                fr = func(xr)
                calls += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fg <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    continue

                # Expansion
                if fr < fg:
                    xe = np.clip(centroid + gamma * (centroid - xw), -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    continue

                # Contraction
                if fr < fvals[-2]:
                    # Outside contraction
                    xc = np.clip(centroid + 0.5 * (xr - centroid), -5, 5)
                else:
                    # Inside contraction
                    xc = np.clip(centroid - 0.5 * (centroid - xw), -5, 5)
                fc = func(xc)
                calls += 1
                if fc < best_f:
                    best_f = fc
                    best_x = xc.copy()

                if fc < min(fr, fvals[-2]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                    continue

                # Shrink
                for i in range(1, dim + 1):
                    simplex[i] = np.clip(xg + sigma * (simplex[i] - xg), -5, 5)
                    fvals[i] = func(simplex[i])
                    calls += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = simplex[i].copy()

            return best_x, best_f

        # Phase 1: Multiple restarts to find good basin
        remaining = self.budget
        restarts = 5
        per_restart = remaining // restarts

        for r in range(restarts):
            if calls >= remaining or time.time() > deadline:
                break

            x0 = rng.uniform(-5, 5, self.dim)
            f0 = func(x0)
            calls += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            max_calls_this = min(remaining - calls, per_restart)
            if max_calls_this > 0:
                nelder_mead(x0, f0, calls + max_calls_this)

        # Phase 2: Final refinement around best point with remaining budget
        if best_x is not None and calls < remaining and time.time() < deadline:
            max_calls_final = remaining - calls
            nelder_mead(best_x, best_f, calls + max_calls_final)

        return best_x, best_f
# EVOLVE-BLOCK-END
