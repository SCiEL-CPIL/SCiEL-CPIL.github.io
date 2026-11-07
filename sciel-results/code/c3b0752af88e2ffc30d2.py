"""Portfolio search optimizer for BBOB f20 (Schwefel) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Portfolio search optimizer for BBOB f20 (Schwefel) on [-5,5]^5.
    Uses a hybrid approach:
    1. Multi-start (1+1)-ES with adaptive step size for global exploration.
    2. Coordinate-descent polish with adaptive step sizes for local convergence.
    3. Explicit deadline guard to stay within wall-clock limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        def time_ok():
            return time.time() < deadline and call_count < budget

        # Phase 1: Multi-start (1+1)-ES global exploration
        # Use ~60% of budget for exploration, ~40% for polish
        explore_budget = int(budget * 0.6)
        polish_budget = budget - explore_budget

        # Number of restarts for multi-start
        n_restarts = 12
        budget_per_restart = explore_budget // n_restarts

        for r in range(n_restarts):
            if not time_ok():
                break

            # Initialize at a random point in the box
            x = rng.uniform(-4, 4, size=dim)
            fx = func(x)
            call_count += 1
            update_best(x, fx)

            # Adaptive step size
            sigma = 1.5
            # Track consecutive failures to adapt sigma
            fail_count = 0
            max_fails = 5

            # Run (1+1)-ES for this restart
            for _ in range(budget_per_restart):
                if not time_ok():
                    break

                # Generate a candidate
                z = rng.standard_normal(dim)
                x_new = x + sigma * z
                x_new = np.clip(x_new, -5, 5)

                fx_new = func(x_new)
                call_count += 1

                if fx_new < fx:
                    # Success: move to new point, increase sigma slightly
                    x = x_new
                    fx = fx_new
                    update_best(x, fx)
                    sigma = min(sigma * 1.1, 5.0)
                    fail_count = 0
                else:
                    # Failure: decrease sigma
                    fail_count += 1
                    sigma = max(sigma * 0.85, 1e-4)

                    # If too many failures, break and restart
                    if fail_count >= max_fails:
                        break

        # Phase 2: Coordinate-descent polish with adaptive step sizes
        if best_x is not None and time_ok():
            x = best_x.copy()
            fx = best_f

            # Adaptive step sizes per dimension
            steps = np.full(dim, 0.5)
            min_step = 1e-6

            # Iterate until budget or time runs out
            improved = True
            while improved and time_ok():
                improved = False
                for d in range(dim):
                    if not time_ok():
                        break

                    # Try positive and negative steps
                    for direction in [1, -1]:
                        if not time_ok():
                            break

                        x_trial = x.copy()
                        x_trial[d] += direction * steps[d]
                        x_trial = np.clip(x_trial, -5, 5)

                        fx_trial = func(x_trial)
                        call_count += 1

                        if fx_trial < fx:
                            x = x_trial
                            fx = fx_trial
                            update_best(x, fx)
                            improved = True
                            # Increase step slightly if it worked
                            steps[d] = min(steps[d] * 1.2, 5.0)
                        else:
                            # Decrease step
                            steps[d] = max(steps[d] * 0.5, min_step)

                    # If step is very small, skip further refinement in this dimension
                    if steps[d] < min_step:
                        continue

        return best_x, best_f
# EVOLVE-BLOCK-END
