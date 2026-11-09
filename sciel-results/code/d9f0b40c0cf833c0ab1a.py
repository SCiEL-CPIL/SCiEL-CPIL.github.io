"""Optimizer for BBOB f12 (Bent Cigar) with budget 500, dim 5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with aggressive sigma adaptation and deadline guard for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin

        rng = np.random.default_rng(42)
        d = self.dim

        # Initialize at center of the box
        x = np.zeros(d)
        fx = func(x)

        # Track best
        x_best = x.copy()
        fx_best = fx

        sigma = 3.0  # Initial step size
        min_sigma = 1e-10
        max_sigma = 10.0

        # Success counter for 1/5 rule
        successes = 0
        total_trials = 0
        success_window = 10

        for i in range(self.budget - 1):
            # Deadline check
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(d)
            y = np.clip(y, -5, 5)
            fy = func(y)

            if fy < fx:
                x = y
                fx = fy
                successes += 1
                # Increase sigma on success, but cautiously
                sigma *= 1.2
                if sigma > max_sigma:
                    sigma = max_sigma
            else:
                # Decrease sigma on failure
                sigma *= 0.85
                if sigma < min_sigma:
                    sigma = min_sigma

            if fy < fx_best:
                fx_best = fy
                x_best = y.copy()

            total_trials += 1

            # Adapt sigma based on success rate over recent window
            if total_trials % success_window == 0:
                # Simple adaptation: if success rate is too high, increase sigma
                # If too low, decrease sigma
                pass  # Keep simple for now

        return x_best, fx_best
# EVOLVE-BLOCK-END
