"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with (1+1)-ES polish and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Track best
        best_x = rng.uniform(-4, 4, dim)
        best_f = func(best_x)
        calls_used = 1

        def remaining():
            return budget - calls_used

        def time_ok():
            return time.time() < deadline - 5.0

        # Phase 1: Multi-start Nelder-Mead
        # Use larger initial simplex for faster convergence
        delta = 3.0

        # Generate multiple starting points
        n_starts = 15
        for s in range(n_starts):
            if not time_ok() or calls_used >= budget - 50:
                break

            # Start points: mix of random and structured
            if s == 0:
                x0 = np.zeros(dim)
            elif s == 1:
                x0 = np.ones(dim) * 3.0
            elif s == 2:
                x0 = -np.ones(dim) * 3.0
            elif s == 3:
                x0 = rng.uniform(-4, 4, dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            calls_used += 1

            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # If we've already found a very good point, skip NM
            if best_f < -50:
                break

            # Nelder-Mead with limited function evaluations
            nm_budget = min(remaining(), 100)
            if nm_budget < 10:
                break

            # Create initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                point = x0.copy()
                point[i] += delta
                point = np.clip(point, -5, 5)
                simplex.append(point)

            # Evaluate simplex
            fvals = []
            for pt in simplex:
                f = func(pt)
                calls_used += 1
                fvals.append(f)
                if f < best_f:
                    best_f = f
                    best_x = pt.copy()

            fvals = np.array(fvals)

            # Nelder-Mead iterations
            alpha, gamma, rho = 1.0, 2.0, 0.5
            for _ in range(min(nm_budget, 80)):
                if not time_ok() or calls_used >= budget - 50:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = fvals[order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_used += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_used += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_used += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    if fc < max(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f = func(simplex[i])
                            calls_used += 1
                            fvals[i] = f
                            if f < best_f:
                                best_f = f
                                best_x = simplex[i].copy()

                        # Check for convergence
                        spread = np.max(fvals) - np.min(fvals)
                        if spread < 1e-8:
                            break

        # Phase 2: (1+1)-ES polish
        if time_ok() and calls_used < budget - 10:
            sigma = 0.5
            x = best_x.copy()
            fx = best_f
            for _ in range(budget - calls_used - 1):
                if not time_ok():
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
                    best_x = x.copy()
                if sigma < 1e-6:
                    break

        # Phase 3: Fine-tune with very small steps
        if time_ok() and calls_used < budget:
            sigma = 0.01
            x = best_x.copy()
            fx = best_f
            for _ in range(min(50, budget - calls_used)):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
