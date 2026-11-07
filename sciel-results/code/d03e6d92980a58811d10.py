"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with deadline guard for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 285  # safety margin before 300s limit
        n = self.dim
        rng = np.random.default_rng(0)

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # Build initial simplex around a random center
        center = rng.uniform(-5.0, 5.0, n)
        simplex = np.zeros((n + 1, n))
        simplex[0] = center
        for i in range(n):
            v = np.zeros(n)
            v[i] = 2.0  # initial step size
            simplex[i + 1] = clip(center + v)

        # Evaluate initial simplex
        fvals = np.array([func(simplex[i]) for i in range(n + 1)])
        calls = n + 1

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        # Track best
        best_idx = int(np.argmin(fvals))
        best_x = simplex[best_idx].copy()
        best_f = fvals[best_idx]

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        while calls < self.budget:
            if time.time() > deadline:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            update_best(simplex[0], fvals[0])

            centroid = simplex[:-1].mean(axis=0)

            # Reflection
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = func(xr)
            calls += 1

            if fr < fvals[0]:
                # Expansion
                xe = clip(centroid + gamma * (centroid - simplex[-1]))
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Simple reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                    fc = func(xc)
                    calls += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            fvals[i] = func(simplex[i])
                            calls += 1
                        if calls >= self.budget:
                            break
                else:
                    # Inside contraction
                    xc = clip(centroid - rho * (centroid - simplex[-1]))
                    fc = func(xc)
                    calls += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            fvals[i] = func(simplex[i])
                            calls += 1
                        if calls >= self.budget:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
