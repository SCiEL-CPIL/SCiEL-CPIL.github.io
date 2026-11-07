"""Optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Island (1+1)-ES with adaptive sigma, restarts, and a local polishing phase."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_budget = self.budget
        dim = self.dim

        # Phase 1: Island exploration with multiple restarts
        # We'll run several (1+1)-ES runs, each with a portion of the budget.
        # Use island model: keep track of best, and restart from best periodically.

        # Strategy: Run multiple (1+1)-ES instances with different initializations.
        # Each gets a chunk of budget. Keep the best overall.
        # Then do a final polishing phase.

        num_islands = 4
        budget_per_island = max(1, (total_budget - 50) // num_islands)  # reserve 50 for polishing
        polishing_budget = total_budget - budget_per_island * num_islands

        for island in range(num_islands):
            if time.time() > deadline:
                break

            # Initialize from best found so far with perturbation, or random for first
            if best_x is not None and island > 0:
                x = np.clip(best_x + rng.normal(0, 1.0, dim), -5, 5)
            else:
                x = rng.uniform(-4, 4, dim)

            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            no_improve = 0
            max_no_improve = 30

            for step in range(budget_per_island):
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    no_improve += 1

                # Adaptive restart within island if stuck
                if no_improve > max_no_improve:
                    # Restart from best with larger perturbation
                    if best_x is not None:
                        x = np.clip(best_x + rng.normal(0, 2.0, dim), -5, 5)
                        fx = func(x)
                    else:
                        x = rng.uniform(-4, 4, dim)
                        fx = func(x)
                    sigma = 3.0
                    no_improve = 0

        # Phase 2: Local polishing around best point
        if best_x is not None and polishing_budget > 0:
            if time.time() < deadline:
                x = best_x.copy()
                fx = best_f
                sigma = 0.5

                for step in range(polishing_budget):
                    if time.time() > deadline:
                        break

                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)

                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.1
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
