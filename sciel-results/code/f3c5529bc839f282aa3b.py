"""Portfolio search optimizer for BBOB f20 (Schwefel) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Multi-start (1+1)-ES with adaptive step size and time-based deadline guard.
    Designed for BBOB f20 (Schwefel) on [-5,5]^5 with budget 500.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Multi-start: 8 diverse initial points
        n_starts = 8
        budget_per_start = max(1, self.budget // n_starts)

        # Generate initial points: mix of random and structured
        initial_points = []
        # Center point
        initial_points.append(np.zeros(self.dim))
        # Random points
        for _ in range(n_starts - 1):
            initial_points.append(rng.uniform(-4, 4, self.dim))

        best_x = None
        best_f = np.inf

        for i, x0 in enumerate(initial_points):
            if time.time() > deadline:
                break

            x = x0.copy()
            fx = func(x)

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            # Adaptive step size parameters
            success_count = 0
            total_steps = 0

            # Use remaining budget for this start
            remaining_budget = self.budget - (i * budget_per_start)
            # But also respect the per-start budget for balance
            steps_for_this = min(budget_per_start, remaining_budget)

            for _ in range(steps_for_this):
                if time.time() > deadline:
                    break

                # Generate perturbation
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)

                total_steps += 1
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                # Adaptive sigma adjustment based on success rate
                if total_steps % 10 == 0:
                    rate = success_count / total_steps
                    if rate > 0.7:
                        sigma *= 1.2
                    elif rate < 0.3:
                        sigma *= 0.8
                    success_count = 0
                    total_steps = 0

                # Ensure sigma stays in reasonable bounds
                sigma = np.clip(sigma, 0.01, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
