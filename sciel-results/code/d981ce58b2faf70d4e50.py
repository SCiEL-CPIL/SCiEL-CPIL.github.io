"""Simulated Annealing optimizer for BBOB f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simulated Annealing with geometric cooling and large initial steps."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = float(func(x))
        best_x = x.copy()
        best_fx = fx

        # Temperature schedule: geometric cooling
        # Start with large step size to explore globally
        sigma_start = 8.0
        sigma_end = 0.01
        n_steps = budget

        # Geometric cooling: sigma_t = sigma_start * (sigma_end/sigma_start)^(t/n_steps)
        # This gives a smooth exponential decay
        ratio = (sigma_end / sigma_start) ** (1.0 / max(n_steps, 1))

        sigma = sigma_start
        temperature = 10.0  # initial "temperature" for acceptance probability scaling

        # We'll track calls to ensure we don't exceed budget
        calls_made = 1

        for step in range(1, n_steps):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate candidate
            direction = rng.standard_normal(dim)
            y = x + sigma * direction
            y = np.clip(y, -5, 5)

            fy = float(func(y))
            calls_made += 1

            # Acceptance criterion
            delta = fy - fx
            if delta < 0:
                # Better solution: always accept
                x = y
                fx = fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                # Worse solution: accept with probability exp(-delta / temperature)
                # Scale temperature with sigma to maintain exploration
                T = temperature * (sigma / sigma_start)
                if T > 1e-10:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        x = y
                        fx = fy

            # Update sigma geometrically
            sigma *= ratio

        return best_x, best_fx
# EVOLVE-BLOCK-END
