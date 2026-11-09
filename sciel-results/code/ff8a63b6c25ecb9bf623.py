"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset.

Uses (1+1)-ES with restarts and Nelder-Mead local polish.
"""
import time
import numpy as np


class Optimizer:
    """(1+1)-ES with restarts and Nelder-Mead polish for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, calls_made, budget, deadline, rng):
        """Nelder-Mead local optimization starting from x0."""
        dim = self.dim
        x = x0.copy()
        f = float(func(x))
        calls_made += 1

        # Initialize simplex
        simplex = [x.copy()]
        simplex_f = [f]
        for i in range(dim):
            x_i = x.copy()
            step = 0.1 if x_i[i] >= 0 else -0.1
            x_i[i] += step
            x_i = np.clip(x_i, -5.0, 5.0)
            f_i = float(func(x_i))
            calls_made += 1
            simplex.append(x_i)
            simplex_f.append(f_i)

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        max_iter = 100

        for _ in range(max_iter):
            if calls_made >= budget or time.time() >= deadline:
                break

            # Sort simplex by fitness
            order = np.argsort(simplex_f)
            simplex = [simplex[i].copy() for i in order]
            simplex_f = [simplex_f[i] for i in order]

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = float(func(x_r))
            calls_made += 1

            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = float(func(x_e))
                calls_made += 1

                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if f_r < simplex_f[-1]:
                    # Outside contraction
                    x_c = centroid + 0.5 * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - 0.5 * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = float(func(x_c))
                calls_made += 1

                if f_c < min(f_r, simplex_f[-1]):
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        x_s = np.clip(simplex[i], -5.0, 5.0)
                        f_s = float(func(x_s))
                        calls_made += 1
                        simplex[i] = x_s
                        simplex_f[i] = f_s

        # Return best point
        best_idx = np.argmin(simplex_f)
        return simplex[best_idx].copy(), simplex_f[best_idx], calls_made

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget
        calls_made = 0

        def still_ok():
            return time.time() < deadline

        # Phase 1: Initial exploration with random points
        n_init = 10
        best_x = np.zeros(dim)
        best_f = np.inf
        for _ in range(n_init):
            x = rng.uniform(-5.0, 5.0, dim)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: (1+1)-ES with restarts
        sigma = 1.0
        no_improve = 0
        threshold = 150  # restart after 150 steps without improvement
        best_x_es = best_x.copy()
        best_f_es = best_f

        while calls_made < budget and still_ok():
            # Generate candidate
            x_new = best_x_es + sigma * rng.standard_normal(dim)
            x_new = np.clip(x_new, -5.0, 5.0)
            f_new = float(func(x_new))
            calls_made += 1

            if f_new < best_f_es:
                best_x_es = x_new.copy()
                best_f_es = f_new
                no_improve = 0
                # Slowly increase sigma on improvement
                sigma *= 1.05
            else:
                no_improve += 1
                # Decrease sigma on no improvement
                sigma *= 0.95

            sigma = np.clip(sigma, 0.01, 5.0)

            # Check for restart
            if no_improve > threshold and calls_made < budget - 50 and still_ok():
                # Restart from a random point
                x_restart = rng.uniform(-5.0, 5.0, dim)
                f_restart = float(func(x_restart))
                calls_made += 1
                if f_restart < best_f_es:
                    best_x_es = x_restart.copy()
                    best_f_es = f_restart
                sigma = 1.0
                no_improve = 0

        # Phase 3: Nelder-Mead polish from best point
        if calls_made < budget - 20 and still_ok():
            best_x_nm, best_f_nm, calls_made = self._nelder_mead(
                func, best_x_es, calls_made, budget, deadline, rng
            )
            if best_f_nm < best_f_es:
                best_x_es = best_x_nm
                best_f_es = best_f_nm

        return best_x_es, float(best_f_es)
