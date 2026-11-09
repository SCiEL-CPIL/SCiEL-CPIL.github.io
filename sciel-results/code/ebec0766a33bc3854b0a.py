"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with origin init, multiplicative sigma adaptation, and multiple restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 315.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Best solution found overall
        best_x = None
        best_f = np.inf
        eval_count = 0

        def try_update(x, fx):
            nonlocal best_x, best_f, eval_count
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Quick diverse sampling to get a good starting point
        # Evaluate several diverse points to find a good region
        sample_points = [
            np.zeros(n),
            rng.uniform(-3, 3, n),
            rng.uniform(-3, 3, n),
            rng.uniform(-4, 4, n) * 0.5,
            rng.uniform(-4, 4, n),
        ]
        
        # Pick the best from diverse samples
        for sp in sample_points:
            if eval_count >= budget - 50:
                break
            sp_clipped = np.clip(sp, -5, 5)
            fsp = func(sp_clipped)
            try_update(sp_clipped, fsp)
        
        # Start from the best point found
        x = best_x.copy()
        fx = best_f

        sigma = 1.5
        sigma_min = 1e-8
        sigma_max = 50.0

        # Multiple restarts with adaptive thresholds
        max_restarts = 4
        for restart in range(max_restarts):
            # Run (1+1)-ES
            if restart == 0:
                sigma = 1.5
            elif restart == 1:
                sigma = 0.8
            elif restart == 2:
                sigma = 0.4
            else:
                sigma = 0.2

            no_improve = 0
            max_no_improve = 60 + restart * 25

            for i in range(budget - eval_count):
                if eval_count >= budget:
                    break
                if time.time() > deadline - 3.0:
                    break

                z = rng.standard_normal(n)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                eval_count += 1

                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.25, sigma_max)
                    no_improve = 0
                else:
                    sigma *= 0.88
                    no_improve += 1

                if sigma < sigma_min or no_improve >= max_no_improve:
                    break

            # If we used all budget or time is up, stop
            if eval_count >= budget or time.time() > deadline - 3.0:
                break

            # Restart: try from a perturbation of best or a random point
            remaining = budget - eval_count
            if remaining < 10:
                break

            if best_x is not None and rng.random() < 0.7:
                # Perturb best solution with moderate noise
                noise_scale = 0.3 + 0.7 * rng.random()
                x = np.clip(best_x + rng.standard_normal(n) * noise_scale, -5, 5)
            else:
                # Try a new random point or origin
                if rng.random() < 0.4:
                    x = np.zeros(n)
                else:
                    x = rng.uniform(-4, 4, n)

            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
