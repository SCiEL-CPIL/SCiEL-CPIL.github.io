"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+lambda)-ES with adaptive step size, time guard, and restarts for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        d = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)

        # Number of restarts: we'll do multiple runs with different starting points
        # Each run gets a portion of the budget
        # Strategy: first run gets most of the budget, subsequent runs get less
        # but we adapt based on time

        # Let's do a multi-start approach with adaptive budget allocation
        # First, try a single (1+lambda)-ES with a good lambda
        # Katsuura is multimodal with many local minima, so we need exploration

        # Strategy:
        # 1. Initial phase: explore with larger steps
        # 2. Refine phase: narrow down with smaller steps
        # 3. If time permits, do restarts from different points

        lambda_ = 4  # number of offspring per generation
        sigma = 3.0  # initial step size
        max_sigma = 8.0
        min_sigma = 1e-6

        # Track success rate for adaptive sigma
        successes = 0
        attempts = 0
        success_threshold = 1.0 / (lambda_ + 1)  # ~1/5 rule generalized

        # Generate initial point
        x = rng.uniform(-4, 4, d)
        fx = func(x)
        calls_made += 1
        best_x = x.copy()
        best_f = fx

        # Main optimization loop
        while calls_made < total_budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate lambda_ offspring
            offspring = np.empty((lambda_, d))
            offspring_f = np.empty(lambda_)

            for i in range(lambda_):
                if calls_made >= total_budget:
                    break
                if time.time() > deadline:
                    break

                # Adaptive sigma adjustment
                if attempts > 10:
                    rate = successes / attempts
                    if rate > success_threshold:
                        sigma *= 1.25
                    elif rate < success_threshold:
                        sigma *= 0.8
                    successes = 0
                    attempts = 0

                sigma = np.clip(sigma, min_sigma, max_sigma)

                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                calls_made += 1
                offspring[i] = y
                offspring_f[i] = fy
                attempts += 1

                if fy < fx:
                    successes += 1

            # Find best offspring
            if lambda_ > 0:
                best_idx = np.argmin(offspring_f[:lambda_])
                if offspring_f[best_idx] < fx:
                    x = offspring[best_idx].copy()
                    fx = offspring_f[best_idx]
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        # If we still have budget and time, do a local refinement with smaller steps
        remaining = total_budget - calls_made
        if remaining > 20 and time.time() < deadline - 10:
            # Local search with very small steps
            local_sigma = 0.1
            x_local = best_x.copy()
            fx_local = best_f

            for _ in range(min(remaining, 100)):
                if time.time() > deadline:
                    break
                y = np.clip(x_local + local_sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx_local:
                    x_local = y
                    fx_local = fy
                    if fx_local < best_f:
                        best_f = fx_local
                        best_x = x_local.copy()
                else:
                    local_sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
