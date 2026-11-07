"""Optimizer for BBOB-style f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        calls = 0
        best_x = None
        best_f = np.inf

        # Multi-restart strategy: use budget across several restarts
        # For f21 with 101 peaks, multiple restarts help find the global min
        # We'll do up to 10 restarts, each getting a portion of the budget
        num_restarts = min(10, budget // 30)
        calls_per_restart = (budget - num_restarts) // num_restarts  # 1 call for initial point per restart

        for r in range(num_restarts):
            # Check deadline
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # New random starting point
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # If we've already exceeded budget, stop
            if calls >= budget:
                break

            # Adaptive (1+1)-ES for this restart
            sigma = 1.5
            # Success counter for 1/5 rule
            successes = 0
            total_trials = 0
            # We'll use a fixed number of steps per restart based on remaining budget
            max_steps = min(calls_per_restart, budget - calls)

            for step in range(max_steps):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    pass  # Keep x, fx

                total_trials += 1

                # 1/5 success rule: adjust sigma every 10 trials
                if total_trials % 10 == 0:
                    if successes >= 2:  # at least 2/10
                        sigma *= 1.2
                    elif successes <= 1:  # at most 1/10
                        sigma *= 0.8
                    successes = 0

                # Adaptive sigma based on progress
                if total_trials % 50 == 0:
                    # If we've been stuck, try a larger jump
                    if sigma < 0.05:
                        sigma = 0.1
                    elif sigma > 3.0:
                        sigma = 2.0

        return best_x, best_f
# EVOLVE-BLOCK-END
