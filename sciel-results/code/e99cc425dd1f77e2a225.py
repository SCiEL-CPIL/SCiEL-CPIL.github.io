"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with random restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            return func(x)

        # Nelder-Mead implementation
        def nelder_mead(func, x0, sigma_init, max_iter):
            n = len(x0)
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(n):
                perturbation = x0.copy()
                perturbation[i] += sigma_init
                perturbation = np.clip(perturbation, -5.0, 5.0)
                simplex.append(perturbation)

            fvals = [call_func(x) for x in simplex]
            best_local = min(fvals)
            best_x_local = simplex[fvals.index(best_local)]

            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = call_func(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = call_func(xe)
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
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = call_func(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = call_func(simplex[i])

                # Track best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_local:
                    best_local = fvals[min_idx]
                    best_x_local = simplex[min_idx]

                # Check convergence
                if max(fvals) - min(fvals) < 1e-10:
                    break

            return best_x_local, best_local

        # Multi-restart strategy
        n_restarts = 5
        sigma_start = 3.0
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # Random starting point
            if r == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4.5, 4.5, dim)

            # Decreasing step size for later restarts
            sigma = sigma_start / (r + 1)

            # Limit iterations per restart
            max_iter_per_restart = max(10, (budget - calls_made) // (n_restarts - r))
            x_best, f_best = nelder_mead(func, x0, sigma, max_iter_per_restart)

            if f_best < best_f:
                best_f = f_best
                best_x = x_best

        return best_x, best_f
# EVOLVE-BLOCK-END
