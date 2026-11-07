"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with diagonal covariance, adaptive step, and restart logic."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        # Parameters
        inc_factor = 1.15
        dec_factor = 0.9
        min_sigma = 1e-8
        max_sigma = 10.0
        no_improve_threshold = 30  # restart if no improvement for this many steps
        restart_sigma = 1.0

        # Initial point: near zero is safe for rotated Rosenbrock
        x = rng.normal(0, 0.5, n)
        fx = func(x)
        calls += 1
        best_x = x.copy()
        best_f = fx

        sigma = np.ones(n) * 0.5
        no_improve = 0
        best_f_so_far = fx

        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x = y
                fx = fy
                no_improve = 0
                # Increase step sizes
                sigma = np.clip(sigma * inc_factor, min_sigma, max_sigma)
                # Update global best
                if fy < best_f:
                    best_f = fy
                    best_x = x.copy()
            else:
                no_improve += 1
                # Decrease step sizes
                sigma = np.clip(sigma * dec_factor, min_sigma, max_sigma)

                # Restart logic: if stuck, try a bigger jump
                if no_improve >= no_improve_threshold:
                    # Try a random restart from a broader distribution
                    # But keep the best point as anchor
                    x = best_x + rng.normal(0, restart_sigma, n)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    if calls >= budget:
                        break
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = np.ones(n) * 0.5
                    no_improve = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
