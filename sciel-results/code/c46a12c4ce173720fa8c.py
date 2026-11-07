"""Optimizer for BBOB-style f13 (Sharp ridge) with adaptive (1+1)-ES and stagnation restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with stagnation-triggered restarts for sharp ridge optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Start at center of box (minimizer is in [-4,4]^5)
        x = np.zeros(d)
        fx = func(x)
        calls_used = 1

        # Adaptive step size
        sigma = 2.0  # Start relatively large to find the ridge quickly
        best_x = x.copy()
        best_f = fx

        # Stagnation tracking
        no_improve_count = 0
        stagnation_threshold = 30  # Restart if no improvement for 30 steps

        # Success rate tracking for sigma adaptation (1/5 rule)
        success_window = []
        window_size = 15

        for step in range(1, budget):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                # Success: move to offspring
                x = y
                fx = fy
                success_window.append(1)
                no_improve_count = 0

                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Failure: stay, record failure
                success_window.append(0)
                no_improve_count += 1

            # Adapt sigma using 1/5 success rule on recent window
            if len(success_window) >= window_size:
                success_rate = sum(success_window[-window_size:]) / window_size
                if success_rate > 0.3:
                    sigma *= 1.1
                elif success_rate < 0.1:
                    sigma *= 0.9
                # Clamp sigma
                sigma = np.clip(sigma, 0.01, 5.0)
                # Keep only recent window
                if len(success_window) > window_size:
                    success_window = success_window[-window_size:]

            # Stagnation restart: if stuck for too long, jump to a new random point
            if no_improve_count >= stagnation_threshold:
                # Try a few random restarts, pick the best one
                best_restart_x = x
                best_restart_f = fx
                for _ in range(3):
                    if time.time() > deadline:
                        break
                    candidate = rng.uniform(-4, 4, d)
                    cf = func(candidate)
                    calls_used += 1
                    if cf < best_restart_f:
                        best_restart_f = cf
                        best_restart_x = candidate

                x = best_restart_x
                fx = best_restart_f
                sigma = 1.5  # Reset step size for new region
                no_improve_count = 0
                success_window = []

                # Update global best if restart found something better
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
