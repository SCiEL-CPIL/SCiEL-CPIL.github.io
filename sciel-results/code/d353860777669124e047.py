"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex optimizer for Schwefel-like f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 300s limit
        calls = 0

        def call_func(x):
            nonlocal calls
            if calls >= budget:
                return None
            calls += 1
            return func(x)

        # Best tracking
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Nelder-Mead simplex method
        def nelder_mead(x0, f0, max_iter=100):
            nonlocal calls, best_x, best_f
            dim = self.dim
            x0 = np.array(x0, dtype=float)
            f0 = float(f0)
            update_best(x0, f0)

            # Build simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            fvals = np.zeros(dim + 1)
            fvals[0] = f0

            for i in range(dim):
                step = 0.5 if x0[i] >= 0 else -0.5
                simplex[i + 1] = x0
                simplex[i + 1][i] += step
                f = call_func(simplex[i + 1])
                if f is None:
                    return
                fvals[i + 1] = f
                update_best(simplex[i + 1], f)

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    return

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Centroid of all but worst
                centroid = simplex[:-1].mean(axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = call_func(xr)
                if fr is None:
                    return
                update_best(xr, fr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    fe = call_func(xe)
                    if fe is None:
                        return
                    update_best(xe, fe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (xr - centroid)
                    fc = call_func(xc)
                    if fc is None:
                        return
                    update_best(xc, fc)
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f = call_func(simplex[i])
                            if f is None:
                                return
                            fvals[i] = f
                            update_best(simplex[i], f)

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    return

        # Multi-start strategy
        rng = np.random.default_rng(42)

        # Initial diverse set of starting points
        n_starts = 8
        starts = []

        # Origin and corners
        starts.append(np.zeros(dim))
        for i in range(dim):
            for sign in [1, -1]:
                pt = np.zeros(dim)
                pt[i] = sign * 4.0
                starts.append(pt)
        starts.append(np.full(dim, 4.0))
        starts.append(np.full(dim, -4.0))

        # Random points
        while len(starts) < n_starts:
            starts.append(rng.uniform(-4, 4, dim))

        # Evaluate and sort starts
        start_results = []
        for s in starts[:n_starts]:
            if calls >= budget or time.time() > deadline:
                break
            f = call_func(s)
            if f is None:
                break
            update_best(s, f)
            start_results.append((f, s))

        # Sort by fitness
        start_results.sort(key=lambda x: x[0])

        # Run Nelder-Mead from best starts
        for f, s in start_results:
            if calls >= budget or time.time() > deadline:
                break
            nelder_mead(s, f, max_iter=80)

        # Final polish: if we have budget left, do a fine-grained local search
        if best_x is not None and calls < budget and time.time() < deadline:
            # Try small perturbations around best
            for _ in range(20):
                if calls >= budget or time.time() > deadline:
                    break
                pert = best_x + 0.1 * rng.standard_normal(dim)
                pert = np.clip(pert, -5, 5)
                f = call_func(pert)
                if f is None:
                    break
                update_best(pert, f)
                if f < best_f:
                    # Continue refining
                    for _ in range(10):
                        if calls >= budget or time.time() > deadline:
                            break
                        pert2 = best_x + 0.05 * rng.standard_normal(dim)
                        pert2 = np.clip(pert2, -5, 5)
                        f2 = call_func(pert2)
                        if f2 is None:
                            break
                        update_best(pert2, f2)

        return best_x, best_f
# EVOLVE-BLOCK-END
