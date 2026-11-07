"""Optimizer for BBOB-style f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid random restart + Nelder-Mead local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 300s kill

        rng = np.random.default_rng(42)

        # Track best solution
        best_x = None
        best_f = np.inf
        calls = 0

        def check_deadline():
            return time.time() > deadline

        def safe_func(x):
            nonlocal calls
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            return float(fx)

        # Phase 1: Random sampling to find promising basins
        # Use ~20% of budget for exploration
        n_explore = max(10, budget // 5)
        for i in range(n_explore):
            if check_deadline() or calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Nelder-Mead local refinement from best point
        # Use remaining budget for local search
        # We'll do multiple restarts of NM if budget allows

        def nelder_mead(x0, max_iter):
            """Nelder-Mead simplex method."""
            nonlocal best_f, best_x, calls
            nonlocal check_deadline

            dim = len(x0)
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [safe_func(x0)]

            # Create other vertices
            for i in range(dim):
                x_new = x0.copy()
                if fvals[0] != 0:
                    x_new[i] += 0.05 * abs(x0[i])
                else:
                    x_new[i] += 0.05
                x_new = np.clip(x_new, -5.0, 5.0)
                simplex.append(x_new)
                fvals.append(safe_func(x_new))

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iter):
                if check_deadline() or calls >= budget:
                    break

                # Sort by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Update global best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

                # Check convergence
                fspread = abs(fvals[-1] - fvals[0])
                if fspread < 1e-12:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = safe_func(x_r)

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = safe_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = safe_func(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = safe_func(simplex[i])

            return best_x, best_f

        # Do local refinement from best point found in exploration
        remaining = budget - calls
        if remaining > 2 * dim and best_x is not None:
            # Split remaining budget among a few local searches
            # First: refine the best point
            n_iter_nm = min(remaining - dim, 200)
            nelder_mead(best_x, n_iter_nm)

            # If we still have budget, try a few more local searches from near-best points
            # Generate a few perturbed starts near the best solution
            if not check_deadline() and calls < budget - 2 * dim:
                n_extra = 3
                for _ in range(n_extra):
                    if check_deadline() or calls >= budget - dim:
                        break
                    # Perturb around best
                    scale = 0.5 * np.ones(dim)
                    x_start = best_x + scale * rng.standard_normal(dim)
                    x_start = np.clip(x_start, -5.0, 5.0)
                    n_iter = min(budget - calls - dim, 100)
                    if n_iter > 0:
                        nelder_mead(x_start, n_iter)

        if best_x is None:
            # Fallback: just return a random point
            best_x = rng.uniform(-5, 5, dim)
            best_f = safe_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
