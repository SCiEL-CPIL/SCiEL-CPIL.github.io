import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s limit

        calls = 0
        best_x = None
        best_f = float('inf')

        def evaluate(x):
            nonlocal calls
            if calls >= budget:
                return None
            calls += 1
            return func(x)

        # Multi-restart (1+1)-ES
        # For Lunacek bi-Rastrigin, try multiple starting points
        # The function has two basins; try points near both

        # Strategy: use several restarts, each with a portion of the budget
        # Allocate budget adaptively based on progress

        n_restarts = 5
        # Initial budget allocation: equal split, then top up best
        budgets_per_restart = [budget // n_restarts] * n_restarts
        # Give a bit more to the first few
        budgets_per_restart[0] += budget % n_restarts

        # Starting points: spread across the domain, biased toward likely basin locations
        # Lunacek bi-Rastrigin has basins near x=0 and x≈3.7 in each dimension
        # But with hidden shifts/rotations, we need to explore broadly
        start_points = []
        # Center
        start_points.append(np.zeros(dim))
        # Near the known secondary basin (approx 3.7 in original coords)
        start_points.append(np.full(dim, 3.0))
        # Random points
        rng = np.random.default_rng(42)
        for _ in range(n_restarts - 2):
            start_points.append(rng.uniform(-4, 4, dim))

        # If more restarts than points, add more random
        while len(start_points) < n_restarts:
            start_points.append(rng.uniform(-4, 4, dim))

        # Run (1+1)-ES for each restart
        for r in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break

            x = start_points[r].copy()
            # Initial step size: moderate
            sigma = 1.0
            budget_r = budgets_per_restart[r]
            calls_at_start = calls

            # (1+1)-ES with 1/5 success rule
            # Track successes in a window
            window = 25
            successes = 0
            total_in_window = 0

            f_x = evaluate(x)
            if f_x is None:
                break
            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()

            for step in range(budget_r):
                if calls >= budget or time.time() > deadline:
                    break

                # Generate candidate
                z = np.random.standard_normal(dim)
                x_new = x + sigma * z
                f_new = evaluate(x_new)
                if f_new is None:
                    break

                # Track success
                if f_new < f_x:
                    successes += 1
                    x = x_new
                    f_x = f_new
                    if f_x < best_f:
                        best_f = f_x
                        best_x = x.copy()
                total_in_window += 1

                # Adapt sigma using 1/5 rule
                if total_in_window >= window:
                    success_rate = successes / total_in_window
                    if success_rate > 0.2:
                        sigma *= 1.05
                    elif success_rate < 0.2:
                        sigma *= 1.0 / 1.05
                    # Reset window
                    successes = 0
                    total_in_window = 0

                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.01, 10.0)

            # After each restart, check if we should allocate more budget to best so far
            # Simple heuristic: if this restart found a new best, give it extra budget
            # But we've already used the budget, so instead we just proceed

        # Final local refinement around best point if budget remains
        if calls < budget and best_x is not None and time.time() < deadline:
            # Use remaining budget for fine-tuning around best
            remaining = budget - calls
            x = best_x.copy()
            f_x = best_f
            sigma = 0.5
            window = 20
            successes = 0
            total_in_window = 0

            for step in range(remaining):
                if calls >= budget or time.time() > deadline:
                    break

                z = np.random.standard_normal(dim)
                x_new = x + sigma * z
                f_new = evaluate(x_new)
                if f_new is None:
                    break

                if f_new < f_x:
                    successes += 1
                    x = x_new
                    f_x = f_new
                    if f_x < best_f:
                        best_f = f_x
                        best_x = x.copy()
                total_in_window += 1

                if total_in_window >= window:
                    success_rate = successes / total_in_window
                    if success_rate > 0.2:
                        sigma *= 1.05
                    elif success_rate < 0.2:
                        sigma *= 1.0 / 1.05
                    successes = 0
                    total_in_window = 0

                sigma = np.clip(sigma, 0.001, 5.0)

        return best_x, best_f
