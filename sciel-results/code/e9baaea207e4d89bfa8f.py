"""Rewritten optimizer for BBOB f23 (Katsuura) using multi-start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead for Katsuura (f23)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        deadline = time.time() + 290.0  # safety margin before 300s

        rng = np.random.default_rng(12345)

        # Track best point and value
        best_x = np.zeros(d)
        best_f = np.inf

        # Generate initial starting points
        # Mix: origin, small random, and larger random
        starts = []
        # Start near origin (Katsuura minimum is often near 0)
        starts.append(np.zeros(d))
        # Small random perturbations around origin
        for _ in range(4):
            starts.append(rng.uniform(-1.5, 1.5, d))
        # Medium random
        for _ in range(3):
            starts.append(rng.uniform(-3, 3, d))
        # Larger random
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, d))

        # Nelder-Mead implementation
        def nelder_mead(func, x0, budget_remaining, deadline, d):
            """Run Nelder-Mead from x0, using up to budget_remaining calls."""
            nonlocal best_x, best_f

            # Initialize simplex
            simplex = [x0.copy()]
            fx_list = [func(x0)]
            if fx_list[0] < best_f:
                best_f = fx_list[0]
                best_x = x0.copy()

            # Add simplex points
            for i in range(d):
                xi = x0.copy()
                step = 0.5 if xi[i] >= 0 else -0.5
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                simplex.append(xi)
                fx_list.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            calls_used = d + 1
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            while calls_used < budget_remaining and time.time() < deadline:
                # Sort simplex by function value
                idx = np.argsort(fx_list)
                simplex = [simplex[i] for i in idx]
                fx_list = [fx_list[i] for i in idx]

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

                if fr < fx_list[0]:
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
                        fx_list[-1] = fe
                    else:
                        simplex[-1] = xr
                        fx_list[-1] = fr
                elif fr < fx_list[-2]:
                    simplex[-1] = xr
                    fx_list[-1] = fr
                else:
                    # Contraction
                    if fr < fx_list[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_used += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    if fc <= fr:
                        simplex[-1] = xc
                        fx_list[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls_used += 1
                            if fi < best_f:
                                best_f = fi
                                best_x = simplex[i].copy()
                            fx_list[i] = fi

            return best_x, best_f

        # Allocate budget across starts
        n_starts = len(starts)
        # Give more budget to earlier starts (they're more promising)
        budgets = []
        remaining = B
        for i in range(n_starts):
            if i == n_starts - 1:
                b = remaining
            else:
                b = max(50, int(remaining * 0.25))
                b = min(b, remaining)
            budgets.append(b)
            remaining -= b

        # Run Nelder-Mead from each start
        for i, start in enumerate(starts):
            if time.time() >= deadline or remaining <= 0:
                break
            b = budgets[i]
            if b < d + 2:
                b = d + 2
            # Run Nelder-Mead
            nelder_mead(func, start, b, deadline, d)
            remaining -= b

        return best_x, best_f
# EVOLVE-BLOCK-END
