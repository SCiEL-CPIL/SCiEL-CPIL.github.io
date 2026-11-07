"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and restarts for bent cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # State for best solution found so far
        best_x = None
        best_f = np.inf

        # We'll do multiple short runs with different strategies
        # Strategy 1: (1+1)-ES with CMA-like adaptation
        # Strategy 2: Coordinate descent / local search
        # Strategy 3: Random restarts with decreasing step

        calls_used = 0

        def check_time():
            return time.time() < deadline

        # Phase 1: (1+1)-ES with adaptive sigma, restart on stagnation
        for restart in range(10):
            if not check_time():
                break

            # Initialize
            if restart == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                # Restart near best found, or random
                if best_x is not None and rng.random() < 0.5:
                    x = best_x + rng.normal(0, 1.0, dim)
                    x = np.clip(x, -5, 5)
                else:
                    x = rng.uniform(-5, 5, dim)

            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            # For bent cigar, we want larger steps in the easy direction
            # Use a simple (1+1)-ES with 1/5 rule

            no_improve = 0
            max_no_improve = 30 + 10 * restart  # more patience on later restarts

            for _ in range(budget - calls_used):
                if not check_time():
                    break

                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    no_improve = 0
                    # Increase step size moderately
                    sigma *= 1.3
                else:
                    no_improve += 1
                    # Decrease step size
                    sigma *= 0.85

                if no_improve >= max_no_improve:
                    break

            if calls_used >= budget:
                break

        # Phase 2: Fine-tuning with very small steps around best
        if best_x is not None and check_time():
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            no_improve = 0
            for _ in range(min(50, budget - calls_used)):
                if not check_time():
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    best_f = fx
                    best_x = x.copy()
                    no_improve = 0
                    sigma *= 1.2
                else:
                    no_improve += 1
                    sigma *= 0.7
                if no_improve >= 15:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
