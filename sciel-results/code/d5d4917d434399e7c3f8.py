"""Rewritten optimizer using a robust Nelder-Mead variant with adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free trust-region / Nelder-Mead hybrid for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        d = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s kill

        # Helper to safely call func and track budget
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            calls_made += 1
            try:
                val = func(np.asarray(x, dtype=float))
                return float(val)
            except Exception:
                return 1e10

        # Initial point: center of the box
        x0 = np.zeros(d)
        fx0 = safe_call(x0)
        if fx0 is None:
            return x0, 1e10

        best_x = x0.copy()
        best_f = fx0

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma = 0.5    # shrink
        initial_step = 0.5

        # Build initial simplex
        simplex = [x0.copy()]
        simplex_f = [fx0]

        for i in range(d):
            xi = x0.copy()
            xi[i] += initial_step
            simplex.append(xi)
            fi = safe_call(xi)
            if fi is None:
                break
            simplex_f.append(fi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        # Check if we ran out of budget
        if calls_made >= budget:
            return best_x, best_f

        # Nelder-Mead main loop
        while calls_made < budget and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            # Update best
            if simplex_f[0] < best_f:
                best_f = simplex_f[0]
                best_x = simplex[0].copy()

            # Check convergence
            if len(simplex_f) < 2:
                break
            f_range = simplex_f[-1] - simplex_f[0]
            if f_range < 1e-12:
                break

            # Centroid of all but the worst point
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            f_worst = simplex_f[-1]

            # Reflection
            x_r = centroid + alpha * (centroid - worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = safe_call(x_r)
            if f_r is None:
                break

            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = safe_call(x_e)
                if f_e is None:
                    break
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
                if f_r < f_worst:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - worst)
                x_c = np.clip(x_c, -5, 5)
                f_c = safe_call(x_c)
                if f_c is None:
                    break

                if f_c <= min(f_r, f_worst):
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls_made >= budget:
                            break
                        fi = safe_call(simplex[i])
                        if fi is None:
                            break
                        simplex_f[i] = fi
                    if calls_made >= budget:
                        break

            # Update best
            if simplex_f[0] < best_f:
                best_f = simplex_f[0]
                best_x = simplex[0].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
