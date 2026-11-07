"""Program for BBOB f18 optimization using I-1/5-ES with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_x = None
        best_f = float('inf')

        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return float(val)

        def i_1_5_es(x0, sigma0, max_calls, rng):
            """
            Run I-1/5-ES from x0 with initial sigma0 for max_calls evaluations.
            Returns best_x, best_f found within this run.
            """
            nonlocal n_calls, best_f, best_x

            if max_calls <= 0 or n_calls >= budget:
                return x0.copy(), best_f

            n = dim
            x = x0.copy()
            sigma = sigma0
            run_best_x = x0.copy()
            run_best_f = best_f

            calls_made = 0
            # Standard 1/5 rule parameters
            # c_sigma = 1/(n+1)
            c_sigma = 1.0 / (n + 1)
            # d_sigma = 1 + 2/max(0, (1)/(n+1)) + c_sigma
            d_sigma = 1 + 2 * max(0, 1.0 / (n + 1)) + c_sigma

            while calls_made < max_calls and n_calls < budget:
                if time.time() > deadline:
                    break

                # Generate candidate
                w = rng.standard_normal(n)
                y = x + sigma * w
                fy = timed_call(y)
                calls_made += 1

                if fy < run_best_f:
                    run_best_f = fy
                    run_best_x = y.copy()

                # Update step size based on success
                if fy < best_f:
                    sigma *= math.exp(c_sigma / d_sigma)
                else:
                    sigma *= math.exp(-c_sigma / d_sigma)

                sigma = np.clip(sigma, 1e-12, 1e12)

                # Move x to y if improvement, else keep x
                if fy < best_f:
                    x = y.copy()

                # Check for convergence (sigma too small)
                if sigma < 1e-10:
                    break

            return run_best_x, run_best_f

        # Strategy:
        # 1. Brief random initialization to find a decent starting point.
        # 2. Multiple restarts of I-1/5-ES.
        # f18 (Schaffers F7) is ill-conditioned and multimodal.
        # I-1/5-ES is robust and simple, good for small dimensions.
        # Restarts help escape local minima.

        rng = np.random.default_rng(42)

        # Phase 1: Initialization
        n_initial = min(20, budget // 10)
        for _ in range(n_initial):
            if n_calls >= budget:
                break
            p = rng.uniform(-4, 4, dim)
            timed_call(p)

        if n_calls >= budget:
            return best_x, best_f

        remaining_budget = budget - n_calls

        # Phase 2: Restarts
        # We allocate budget to restarts.
        # Each restart runs until stagnation (sigma collapses) or budget runs out.
        # We will run restarts sequentially.
        # To ensure diversity, we mix starting points:
        # - Best found so far (perturbed)
        # - Random points

        # Let's try 5 restarts.
        n_restarts = 5
        # Budget per restart is not fixed strictly, but we let I-1/5-ES run until it converges or time/calls run out.
        # However, to avoid one run eating everything if it doesn't converge fast, we can cap it.
        # But I-1/5-ES usually converges quickly in 5D.
        # Let's just loop and let the internal logic handle termination.

        for i in range(n_restarts):
            if n_calls >= budget or time.time() > deadline:
                break

            # Choose starting point
            if i == 0:
                # Start from current best
                x0 = best_x.copy()
                sigma0 = 1.0
            elif i == 1:
                # Perturb best
                x0 = best_x + 1.0 * rng.standard_normal(dim)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 0.5
            else:
                # Random restart
                x0 = rng.uniform(-4, 4, dim)
                sigma0 = 1.0

            # Run I-1/5-ES
            # Give it a generous max_calls, it will stop early if converged
            max_calls_for_run = min(remaining_budget - n_calls, 200)
            i_1_5_es(x0, sigma0, max_calls_for_run, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
