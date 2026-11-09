"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with restarts and time guard for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        total_calls = 0

        rng = np.random.default_rng(42)

        # Use multiple restarts to handle the sharp ridge structure
        # f13 has a narrow ridge, so we need good exploration along the ridge direction

        max_restarts = 8
        calls_per_restart = self.budget // max_restarts
        # But we'll adapt: use more calls if we're making progress

        for restart in range(max_restarts):
            if total_calls >= self.budget:
                break
            if time.time() > deadline:
                break

            # Initialize: random point in the box
            # For sharp ridge, the optimum is often near the center or shifted
            # Try a mix of center and random
            if restart == 0:
                x = np.zeros(self.dim)
            elif restart == 1:
                x = rng.uniform(-2, 2, self.dim)
            else:
                x = rng.uniform(-5, 5, self.dim)

            fx = func(x)
            total_calls += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive sigma for (1+1)-ES
            sigma = 1.0
            stall = 0
            max_stall = 30

            for step in range(calls_per_restart):
                if total_calls >= self.budget:
                    break
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.3
                    stall = 0
                else:
                    sigma *= 0.85
                    stall += 1

                # If stalled too long, restart within this segment
                if stall > max_stall:
                    break

            # If we used fewer calls than allocated, redistribute
            # Actually, let's just let the loop handle it naturally

        return best_x, best_f
# EVOLVE-BLOCK-END
