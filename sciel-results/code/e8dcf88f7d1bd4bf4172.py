"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with multi-start and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s limit
        dim = self.dim
        budget = self.budget

        calls = 0

        def counted_func(x):
            nonlocal calls
            calls += 1
            x = np.asarray(x, dtype=np.float64).reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        def nelder_mead(func, x0, initial_step=0.5, max_iter=1000):
            nonlocal calls
            n = dim
            simplex = [np.array(x0, dtype=np.float64)]
            fvals = [func(simplex[0])]
            for i in range(n):
                v = simplex[0].copy()
                v[i] += initial_step
                v = np.clip(v, -5.0, 5.0)
                simplex.append(v)
                fvals.append(func(v))

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

            for _ in range(max_iter):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if abs(fvals[-1] - fvals[0]) < 1e-12:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), -5.0, 5.0)
                f_r = func(x_r)

                if f_r < fvals[0]:
                    x_e = np.clip(centroid + gamma * (x_r - centroid), -5.0, 5.0)
                    f_e = func(x_e)
                    if f_e < f_r:
                        simplex[-1], fvals[-1] = x_e, f_e
                    else:
                        simplex[-1], fvals[-1] = x_r, f_r
                elif f_r < fvals[-2]:
                    simplex[-1], fvals[-1] = x_r, f_r
                else:
                    if f_r < fvals[-1]:
                        x_c = np.clip(centroid + rho * (x_r - centroid), -5.0, 5.0)
                    else:
                        x_c = np.clip(centroid - rho * (simplex[-1] - centroid), -5.0, 5.0)
                    f_c = func(x_c)
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1], fvals[-1] = x_c, f_c
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5.0, 5.0)
                            fvals[i] = func(simplex[i])

            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        rng = np.random.default_rng(123)
        best_x, best_f = None, np.inf

        # Multi-start with diverse initial points
        num_starts = 8
        budget_per_start = max(20, (budget - 100) // num_starts)

        start_points = [
            np.zeros(dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(1, 4, dim),
            rng.uniform(-4, -1, dim),
            rng.uniform(-5, 5, dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-5, 5, dim),
        ]

        for s in range(num_starts):
            if calls >= budget - 50 or time.time() > deadline:
                break
            x0 = np.clip(start_points[s], -5.0, 5.0)
            x_opt, f_opt = nelder_mead(counted_func, x0, initial_step=1.0, max_iter=budget_per_start)
            if f_opt < best_f:
                best_f, best_x = f_opt, x_opt.copy()

        # Local refinement
        if best_x is not None and calls < budget - 10 and time.time() < deadline:
            x_ref, f_ref = nelder_mead(counted_func, best_x, initial_step=0.1, max_iter=budget - calls - 5)
            if f_ref < best_f:
                best_f, best_x = f_ref, x_ref.copy()

        if calls == 0:
            counted_func(np.zeros(dim))

        return best_x, best_f
# EVOLVE-BLOCK-END
