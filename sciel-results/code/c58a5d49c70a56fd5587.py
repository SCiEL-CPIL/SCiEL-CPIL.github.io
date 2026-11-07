"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive restarts, multi-point exploration, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 285s safety margin (330s limit)
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best-so-far
        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def try_update(x, f):
            nonlocal best_x, best_f, calls_made
            if f < best_f:
                best_x, best_f = x.copy(), f

        # Phase 1: Multi-point exploration (use ~10% of budget or at least 20)
        n_explore = max(20, min(50, budget // 10))
        for i in range(n_explore):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            try_update(x, fx)

        # Phase 2: (1+1)-ES with adaptive restarts from best-so-far
        # Start from best found in exploration
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        remaining = budget - calls_made

        # Track stall: no improvement in last N steps
        stall_counter = 0
        stall_threshold = 15
        restart_count = 0
        max_restarts = 10

        for step in range(remaining):
            if not time_ok():
                break

            # Adaptive restart: if stuck, jump from best with larger sigma
            if stall_counter >= stall_threshold and restart_count < max_restarts:
                # Restart from best with increased sigma
                sigma = max(sigma * 2.0, 0.5)
                x = best_x.copy()
                fx = best_f
                stall_counter = 0
                restart_count += 1
                # Small perturbation to escape exact same point
                x = np.clip(x + 0.1 * rng.standard_normal(dim), -5, 5)
                fx = func(x)
                calls_made += 1
                try_update(x, fx)
                continue

            # Standard (1+1)-ES step
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                stall_counter = 0
                try_update(x, fx)
            else:
                sigma *= 0.9
                stall_counter += 1

            # Early termination if we're very close to a good solution
            # and have used significant budget
            if calls_made > budget * 0.8 and best_f < 1e-3:
                # Continue but with smaller steps
                sigma = min(sigma, 0.01)

        return best_x, best_f
# EVOLVE-BLOCK-END
