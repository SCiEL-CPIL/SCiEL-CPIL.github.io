"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and adaptive restarts for f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin

        rng = np.random.default_rng(12345)

        # Start with a random point
        x = rng.uniform(-5, 5, dim)
        f = func(x)
        calls = 1

        best_x = x.copy()
        best_f = f

        # 1/5 rule parameters
        sigma = 1.0
        n_success = 0
        n_trials = 0
        window = 10  # evaluate every 10 steps to adjust sigma

        # Restart parameters
        no_improve = 0
        restart_threshold = 30  # restart if no improvement for this many evals
        max_restarts = 8

        restarts = 0

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate a trial step
            z = rng.standard_normal(dim)
            x_trial = x + sigma * z
            x_trial = np.clip(x_trial, -5, 5)
            f_trial = func(x_trial)
            calls += 1

            n_trials += 1

            if f_trial < f:
                # Accept
                x = x_trial
                f = f_trial
                n_success += 1
                no_improve = 0

                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            else:
                # Reject
                no_improve += 1

            # Adjust sigma every `window` trials
            if n_trials % window == 0:
                rate = n_success / window
                if rate > 0.2:
                    sigma *= 1.5
                elif rate < 0.2:
                    sigma /= 1.5
                sigma = np.clip(sigma, 1e-8, 5.0)
                n_success = 0
                n_trials = 0

            # Check for restart
            if no_improve >= restart_threshold and restarts < max_restarts:
                restarts += 1
                # Perturb current best or start fresh
                if rng.random() < 0.5:
                    # Local perturbation around best
                    x = best_x + rng.normal(0, 1.0, dim)
                    x = np.clip(x, -5, 5)
                else:
                    # Fresh random start
                    x = rng.uniform(-5, 5, dim)

                f = func(x)
                calls += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

                # Reset counters
                no_improve = 0
                sigma = 1.0
                n_success = 0
                n_trials = 0
                restart_threshold = min(restart_threshold + 10, 60)

        return best_x, best_f
# EVOLVE-BLOCK-END
