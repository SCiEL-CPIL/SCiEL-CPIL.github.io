"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Sobol quasi-random global search with adaptive variance reduction
    and dynamic budget allocation across multiple (1+1)-ES restarts.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        calls_used = 0

        def f(x):
            nonlocal calls_used
            calls_used += 1
            return func(x)

        # --- Sobol quasi-random sequence for systematic domain coverage ---
        def sobol_sequence(n, dim, seed=42):
            """Generate n Sobol quasi-random points in [0,1]^dim using a simple implementation."""
            rng = np.random.default_rng(seed)
            # Use a simple Halton-like sequence as a lightweight Sobol substitute
            # or use scipy's Sobol if available
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
                points = sampler.random(n)
                return points
            except ImportError:
                # Fallback: use Latin hypercube sampling
                points = np.zeros((n, dim))
                for d in range(dim):
                    perm = rng.permutation(n)
                    points[:, d] = (perm + 0.5) / n
                return points

        # Generate Sobol points mapped to [-5, 5]^dim
        n_sobol = min(200, budget // 5)  # Use up to 200 Sobol points or 1/5 of budget
        if n_sobol < 10:
            n_sobol = 10

        sobol_points = sobol_sequence(n_sobol, dim)
        sobol_points = sobol_points * 10.0 - 5.0  # Map [0,1] to [-5,5]

        # --- Phase 1: Evaluate Sobol points to find promising regions ---
        best_x = None
        best_f = np.inf

        for pt in sobol_points:
            if calls_used >= budget or time.time() > deadline:
                break
            pt_clipped = np.clip(pt, -5, 5)
            fp = f(pt_clipped)
            if fp < best_f:
                best_f = fp
                best_x = pt_clipped.copy()

        if best_x is None:
            # Fallback: start from origin if no Sobol points were evaluated
            best_x = np.zeros(dim)
            best_f = f(best_x)

        # --- Phase 2: Adaptive local refinement with dynamic budget allocation ---
        # Use (1+1)-ES with sigma adaptation and stagnation detection
        def local_es(x0, fx0, max_calls, sigma_init=2.0):
            """
            (1+1)-ES with 1/5 success rule and stagnation reset.
            Returns (best_x, best_f, calls_used_in_this_es).
            """
            x = x0.copy()
            fx = fx0
            sigma = sigma_init
            rng = np.random.default_rng(0)
            calls_local = 0
            stagnant = 0
            best_local_x = x.copy()
            best_local_f = fx
            no_improve = 0

            for _ in range(max_calls):
                if calls_used + calls_local >= budget or time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = f(y)
                calls_local += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                    stagnant = 0
                    no_improve = 0
                    if fx < best_local_f:
                        best_local_f = fx
                        best_local_x = x.copy()
                else:
                    sigma *= 0.87
                    stagnant += 1
                    no_improve += 1

                # Stagnation detection: if no improvement for 30 steps, reset sigma
                if no_improve >= 30:
                    sigma = max(sigma * 2.0, 0.5)
                    no_improve = 0

            return best_local_x, best_local_f, calls_local

        # Allocate remaining budget to local refinement
        remaining_budget = budget - calls_used
        if remaining_budget > 0:
            # Use 80% of remaining for main refinement, 20% for potential second restart
            main_budget = int(remaining_budget * 0.8)
            restart_budget = remaining_budget - main_budget

            # Main refinement
            x_best, f_best, _ = local_es(best_x, best_f, main_budget, sigma_init=2.0)

            # If we have budget left, try a restart from a different region
            if restart_budget > 10 and time.time() < deadline:
                # Pick a second-best starting point from Sobol evaluation
                # or use a random point far from current best
                rng_restart = np.random.default_rng(123)
                x_restart = rng_restart.uniform(-4.5, 4.5, dim)
                x_restart = np.clip(x_restart, -5, 5)
                fx_restart = f(x_restart)

                x_restart, f_restart, _ = local_es(x_restart, fx_restart, restart_budget, sigma_init=3.0)

                if f_restart < f_best:
                    x_best, f_best = x_restart, f_restart

        return x_best, f_best
# EVOLVE-BLOCK-END
