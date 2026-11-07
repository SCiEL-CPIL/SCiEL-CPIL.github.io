"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid global-local optimizer: random restarts + Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        def nelder_mead(x0, fx0, max_iter):
            """Nelder-Mead with adaptive step size."""
            nonlocal best_x, best_f
            n = dim
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [fx0]
            for i in range(n):
                xi = x0.copy()
                step = 0.1 if i == 0 else 0.05
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                simplex.append(xi)
                fvals.append(fi)
                if fi < best_f:
                    best_f, best_x = fi, xi.copy()

            alpha, gamma, rho = 1.0, 2.0, 0.5

            for _ in range(max_iter):
                if time.time() > deadline:
                    return

                # Sort by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Update global best
                if fvals[0] < best_f:
                    best_f, best_x = fvals[0], simplex[0].copy()

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                if fr < best_f:
                    best_f, best_x = fr, xr.copy()

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    if fe < best_f:
                        best_f, best_x = fe, xe.copy()
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                elif fr < fvals[-2]:
                    simplex[-1], fvals[-1] = xr, fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    if fc < best_f:
                        best_f, best_x = fc, xc.copy()
                    if fc < fr:
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + rho * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            if fvals[i] < best_f:
                                best_f, best_x = fvals[i], simplex[i].copy()

                # Check convergence
                fspan = max(fvals) - min(fvals)
                if fspan < 1e-10:
                    break

            return

        # Phase 1: Global exploration with random restarts
        # Use ~30% of budget for exploration, 70% for local refinement
        explore_budget = max(50, budget // 3)
        exploit_budget = budget - explore_budget

        # Try several random starting points
        n_restarts = min(10, explore_budget // 5)
        start_points = []
        for _ in range(n_restarts):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            start_points.append((fx, x))
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        # Sort by function value and pick top candidates
        start_points.sort(key=lambda t: t[0])
        n_local = min(5, len(start_points))

        # Phase 2: Local refinement with Nelder-Mead on best candidates
        calls_made = n_restarts
        for i in range(n_local):
            if time.time() > deadline:
                break
            fx, x = start_points[i]
            remaining = budget - calls_made
            if remaining <= 10:
                break
            max_iter = max(20, remaining // (n_local - i))
            nelder_mead(x, fx, max_iter)
            calls_made = budget  # approximate

        # Phase 3: If budget remains, do more refinement from best point
        if best_x is not None and time.time() < deadline:
            remaining = budget - n_restarts - 5 * 40  # rough estimate
            if remaining > 50:
                nelder_mead(best_x, best_f, min(200, remaining))

        return best_x, best_f
# EVOLVE-BLOCK-END
