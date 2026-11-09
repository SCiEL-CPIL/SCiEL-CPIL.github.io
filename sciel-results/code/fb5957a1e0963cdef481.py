"""Nelder-Mead simplex with multi-start for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex for 5-dim ill-conditioned functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.asarray(x, dtype=float)
            np.clip(x, -5, 5, out=x)
            f_val = func(x)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead(x0, max_evals, initial_step=1.0):
            nonlocal calls_made, best_x, best_f
            if calls_made >= max_evals:
                return

            # Initialize simplex: x0 plus dim perturbed points
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(dim):
                simplex[i + 1] = x0.copy()
                simplex[i + 1, i] += initial_step
            np.clip(simplex, -5, 5, out=simplex)

            # Evaluate initial simplex
            f_values = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls_made >= max_evals:
                    return
                f_values[i] = evaluate(simplex[i])

            # Nelder-Mead parameters
            alpha = 1.0    # reflection
            gamma = 2.0    # expansion
            rho = 0.5      # contraction
            sigma = 0.5    # shrink

            # Sort simplex by function value
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]

            remaining = max_evals - calls_made
            while remaining > 0 and time.time() < deadline:
                # Centroid of all but worst point
                centroid = simplex[:-1].mean(axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                np.clip(x_r, -5, 5, out=x_r)
                if calls_made >= max_evals:
                    break
                f_r = evaluate(x_r)
                remaining -= 1

                if f_r < f_values[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    np.clip(x_e, -5, 5, out=x_e)
                    if calls_made >= max_evals:
                        break
                    f_e = evaluate(x_e)
                    remaining -= 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_values[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    np.clip(x_c, -5, 5, out=x_c)
                    if calls_made >= max_evals:
                        break
                    f_c = evaluate(x_c)
                    remaining -= 1
                    if f_c < min(f_r, f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            np.clip(simplex[i], -5, 5, out=simplex[i])
                        if calls_made >= max_evals:
                            break
                        f_values[i] = evaluate(simplex[i])
                        remaining -= 1

                # Re-sort
                order = np.argsort(f_values)
                simplex = simplex[order]
                f_values = f_values[order]

                # Check convergence
                if np.max(f_values) - np.min(f_values) < 1e-10:
                    break

        # Multi-start strategy
        # Start 1: center of box
        x0 = np.zeros(dim)
        f0 = evaluate(x0)
        
        # Use remaining budget for Nelder-Mead
        # Split budget: give most to first run, some to subsequent runs
        # First run gets 400 evals, subsequent runs get 50 each
        first_budget = 400
        restart_budget = 50
        
        nelder_mead(x0, first_budget, initial_step=2.0)
        
        # Additional restarts from best point with different perturbations
        num_restarts = (budget - calls_made) // restart_budget
        for r in range(num_restarts):
            if calls_made >= budget or time.time() >= deadline:
                break
            # Perturb best point
            x_start = best_x + 1.0 * rng.standard_normal(dim)
            np.clip(x_start, -5, 5, out=x_start)
            nelder_mead(x_start, restart_budget, initial_step=0.5)
        
        # If still have budget, do more local refinement
        while calls_made < budget and time.time() < deadline:
            x_start = best_x + 0.1 * rng.standard_normal(dim)
            np.clip(x_start, -5, 5, out=x_start)
            nelder_mead(x_start, min(20, budget - calls_made), initial_step=0.1)

        return best_x, best_f
# EVOLVE-BLOCK-END
