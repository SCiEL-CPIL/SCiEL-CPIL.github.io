# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead with random warm-up for BBOB f23 (Katsuura)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Random warm-up to get a good starting point
        n_random = min(30, budget // 10)
        rng = np.random.default_rng(42)
        for _ in range(n_random):
            x = rng.uniform(-5, 5, size=dim)
            evaluate(x)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = evaluate(best_x)

        # Phase 2: Multi-start Nelder-Mead
        n_starts = 8
        nm_budget_per_start = (budget - calls_made) // n_starts

        for start in range(n_starts):
            if calls_made >= budget:
                break

            # Initialize simplex around current best
            center = best_x.copy()
            simplex = [center.copy()]
            for i in range(dim):
                x_i = center.copy()
                x_i[i] += 0.5 * (1.0 if i % 2 == 0 else -1.0)
                x_i = np.clip(x_i, -5, 5)
                simplex.append(x_i)

            f_vals = [evaluate(simplex[0])]
            for i in range(1, dim + 1):
                f_vals.append(evaluate(simplex[i]))

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            max_iter = nm_budget_per_start - (dim + 1)
            if max_iter < 10:
                max_iter = 10

            for _ in range(max_iter):
                if calls_made >= budget:
                    break

                # Sort simplex by fitness
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = evaluate(x_ref)

                if f_ref < f_vals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = evaluate(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_vals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_vals[-1] = f_ref
                elif f_ref < f_vals[-2]:
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_vals[-1]:
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        x_con = centroid - rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = evaluate(x_con)

                    if f_con <= f_ref:
                        simplex[-1] = x_con
                        f_vals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls_made < budget:
                                f_vals[i] = evaluate(simplex[i])
                            else:
                                f_vals[i] = np.inf
                                break

            # Update best from simplex
            for i in range(dim + 1):
                if f_vals[i] < best_f:
                    best_f = f_vals[i]
                    best_x = simplex[i].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
