"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, followed by Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0

        rng = np.random.default_rng(0)

        # Start at origin: the hidden shift is in [-4,4]^5, so origin is a reasonable prior
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        sigma = 2.0

        # Phase 1: (1+1)-ES with 1/5 success rule for global exploration and convergence
        # Reserve ~80 calls for final Nelder-Mead polish
        es_budget = budget - 80

        while calls < es_budget:
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.87

        # Phase 2: Nelder-Mead polish for high-precision refinement
        remaining = budget - calls
        if remaining > 10 and time.time() < deadline:
            try:
                # Build initial simplex around current best
                simplex = [best_x.copy()]
                simplex_f = [best_f]

                scale = 0.1
                for j in range(dim):
                    x_j = best_x.copy()
                    x_j[j] += scale
                    if x_j[j] > 5:
                        x_j[j] = best_x[j] - scale
                    f_j = func(x_j)
                    calls += 1
                    simplex.append(x_j)
                    simplex_f.append(f_j)

                # Nelder-Mead iterations
                alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
                max_iter = remaining - dim - 1

                for _ in range(max_iter):
                    if calls >= budget or time.time() > deadline:
                        break

                    # Sort simplex by function value
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]

                    # Centroid of all but worst point
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5, 5)
                    f_r = func(x_r)
                    calls += 1

                    if f_r < simplex_f[0]:
                        # Expansion
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        f_e = func(x_e)
                        calls += 1
                        if f_e < f_r:
                            simplex[-1] = x_e
                            simplex_f[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                    elif f_r < simplex_f[-2]:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    else:
                        # Contraction
                        x_c = centroid + rho * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        calls += 1
                        if f_c < simplex_f[-1]:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for k in range(1, len(simplex)):
                                simplex[k] = (simplex[0] + simplex[k]) / 2.0
                                simplex_f[k] = func(simplex[k])
                                calls += 1
                                if calls >= budget or time.time() > deadline:
                                    break

                    # Track best
                    min_f = min(simplex_f)
                    if min_f < best_f:
                        best_idx = np.argmin(simplex_f)
                        best_f = min_f
                        best_x = simplex[best_idx].copy()

            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
