"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill

        calls = 0
        best_x = None
        best_f = np.inf

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: (1+1)-ES for global search ---
        rng = np.random.default_rng(0)
        x = rng.uniform(-5.0, 5.0, self.dim)
        fx = eval_func(x)
        sigma = 2.0

        es_budget = int(self.budget * 0.85)  # Use 85% of budget for ES
        while calls < es_budget:
            if time.time() > deadline - 30:  # Reserve time for NM
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5.0, 5.0)
            fy = eval_func(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.85
            sigma = max(sigma, 0.01)

        # --- Phase 2: Nelder-Mead refinement ---
        if best_x is None:
            return x, fx

        x0 = best_x.copy()
        # Initialize simplex around best point
        simplex = np.zeros((self.dim + 1, self.dim))
        simplex[0] = x0
        step = 0.1
        for i in range(1, self.dim + 1):
            simplex[i] = x0.copy()
            simplex[i][i - 1] += step

        simplex_f = np.array([eval_func(xi) for xi in simplex])

        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5

        while calls < self.budget:
            if time.time() > deadline:
                break

            # Sort simplex
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]

            # Convergence check
            if abs(simplex_f[0] - simplex_f[1]) < 1e-10:
                break

            centroid = np.mean(simplex[:self.dim], axis=0)

            # Reflection
            x_r = centroid - alpha * (centroid - simplex[-1])
            f_r = eval_func(x_r)

            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                f_e = eval_func(x_e)
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
                if f_r < simplex_f[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                    f_c = eval_func(x_c)
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex_f[i] = eval_func(simplex[i])
                else:
                    x_c = centroid - rho * (centroid - simplex[-1])
                    f_c = eval_func(x_c)
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex_f[i] = eval_func(simplex[i])

        return best_x, best_f
# EVOLVE-BLOCK-END
