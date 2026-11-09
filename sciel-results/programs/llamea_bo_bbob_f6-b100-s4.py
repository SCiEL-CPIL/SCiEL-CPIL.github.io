"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead based local search with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Initial random start
        x = rng.uniform(-4, 4, self.dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx

        # Nelder-Mead parameters
        initial_simplex_size = 0.5

        # Try multiple restarts if budget allows
        max_restarts = 5
        budget_per_restart = max(50, (self.budget - 1) // max_restarts)

        for restart in range(max_restarts):
            # Build initial simplex
            simplex = [x.copy()]
            for i in range(self.dim):
                point = x.copy()
                point[i] += initial_simplex_size
                point = np.clip(point, -5, 5)
                simplex.append(point)

            simplex_f = [fx] + [func(p) for p in simplex[1:]]

            # Track calls in this restart
            calls_in_restart = 1 + self.dim  # initial point + simplex construction

            # Nelder-Mead iteration
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

            for _ in range(budget_per_restart):
                # Check time
                if time.time() > deadline:
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
                calls_in_restart += 1

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_in_restart += 1
                    if f_e < f_r:
                        simplex[-1], simplex_f[-1] = x_e, f_e
                    else:
                        simplex[-1], simplex_f[-1] = x_r, f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1], simplex_f[-1] = x_r, f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_in_restart += 1
                    if f_c < simplex_f[-1]:
                        simplex[-1], simplex_f[-1] = x_c, f_c
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = func(simplex[i])
                            calls_in_restart += 1
                        if calls_in_restart >= budget_per_restart:
                            break

                # Update best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_fx:
                    best_fx = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()

                # Check for convergence
                if calls_in_restart >= budget_per_restart:
                    break

            # For next restart, start from current best
            x = best_x.copy()
            fx = best_fx

            # If we've used most of our budget, stop
            if calls_in_restart >= budget_per_restart and restart < max_restarts - 1:
                # Try a smaller step for next restart
                initial_simplex_size *= 0.5

        # Final polish with scipy minimize if time allows
        if time.time() < deadline - 30:
            try:
                result = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if result.fun < best_fx:
                    best_x, best_fx = result.x.copy(), result.fun
            except:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
