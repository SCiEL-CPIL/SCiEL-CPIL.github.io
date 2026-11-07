"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive step size and restarts for 5D BBOB f6."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(42)

        dim = self.dim
        box_lo, box_hi = -5.0, 5.0

        calls_made = 0
        x_best = np.zeros(dim)
        f_best = float("inf")

        def _clip(x):
            return np.clip(x, box_lo, box_hi)

        def _evaluate(x):
            nonlocal calls_made, f_best, x_best
            y = _clip(x)
            f = func(y)
            calls_made += 1
            if f < f_best:
                f_best = f
                x_best = y.copy()
            return y, f

        # Nelder-Mead implementation with adaptive step size
        def nelder_mead(x0, f0, budget_remaining):
            nonlocal calls_made, f_best, x_best
            if budget_remaining <= 0:
                return x0, f0

            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [f0]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.1
                xi = _clip(xi)
                yi, fi = _evaluate(xi)
                simplex.append(yi)
                fvals.append(fi)
                budget_remaining -= 1
                if budget_remaining <= 0:
                    break

            # Sort simplex
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            while budget_remaining > 0 and time.time() < deadline:
                # If the best hasn't changed much, shrink
                if fvals[-1] - fvals[0] < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = _clip(x_r)
                _, f_r = _evaluate(x_r)
                budget_remaining -= 1

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = _clip(x_e)
                    _, f_e = _evaluate(x_e)
                    budget_remaining -= 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = _clip(x_c)
                    _, f_c = _evaluate(x_c)
                    budget_remaining -= 1
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = _clip(simplex[i])
                            _, f_i = _evaluate(simplex[i])
                            budget_remaining -= 1
                            fvals[i] = f_i
                            if budget_remaining <= 0:
                                break

                # Sort simplex
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

            return simplex[0], fvals[0]

        # Multi-restart strategy
        n_restarts = 5
        budget_per_restart = self.budget // n_restarts

        for r in range(n_restarts):
            if calls_made >= self.budget or time.time() > deadline:
                break

            # Initial point: first restart at origin, subsequent at best found + noise
            if r == 0:
                x0 = np.zeros(dim)
            else:
                x0 = x_best + rng.standard_normal(dim) * 1.0
                x0 = _clip(x0)

            y0, f0 = _evaluate(x0)
            budget_per_restart = min(budget_per_restart, self.budget - calls_made)
            if budget_per_restart <= 0:
                break

            x_opt, f_opt = nelder_mead(x0, f0, budget_per_restart)

        return x_best, f_best
# EVOLVE-BLOCK-END
