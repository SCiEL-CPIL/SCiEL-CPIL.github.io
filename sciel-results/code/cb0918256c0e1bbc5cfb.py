"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate-descent hybrid with adaptive step sizes and random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        deadline = time.time() + 280.0  # safety margin before 330s kill
        calls = 0
        best_x = None
        best_f = np.inf

        def count_call():
            nonlocal calls
            calls += 1

        # Phase 1: Multi-start coordinate descent with adaptive steps
        # We'll do several restarts, each with a budget of ~80-100 calls
        n_restarts = 6
        budget_per_restart = max(50, (self.budget - 10) // n_restarts)

        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            if calls >= self.budget:
                break

            # Initialize: center for first restart, random for others
            if restart == 0:
                x = np.zeros(self.dim)
            else:
                x = rng.uniform(-4.5, 4.5, self.dim)

            fx = func(x)
            count_call()
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive step sizes per dimension
            steps = np.ones(self.dim) * 1.5
            max_no_improve = 10

            no_improve = 0
            max_iter = budget_per_restart

            for it in range(max_iter):
                if time.time() > deadline:
                    break
                if calls >= self.budget:
                    break

                # Cycle through dimensions
                dim_idx = it % self.dim
                improved_this_pass = False

                # Try both directions
                for sign in [1.0, -1.0]:
                    if calls >= self.budget:
                        break
                    y = x.copy()
                    y[dim_idx] = np.clip(y[dim_idx] + sign * steps[dim_idx], -5.0, 5.0)
                    fy = func(y)
                    count_call()

                    if fy < fx:
                        x = y
                        fx = fy
                        steps[dim_idx] *= 1.3
                        improved_this_pass = True
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        no_improve = 0
                    else:
                        steps[dim_idx] *= 0.7
                        if steps[dim_idx] < 0.01:
                            steps[dim_idx] = 0.01

                if improved_this_pass:
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= max_no_improve:
                        # Small random perturbation to escape
                        x = np.clip(x + rng.normal(0, 0.5, self.dim), -5.0, 5.0)
                        fx = func(x)
                        count_call()
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        no_improve = 0
                        steps = np.maximum(steps, 0.5)

        # Phase 2: Fine-tuning around best point with smaller steps
        if best_x is not None and calls < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            steps = np.ones(self.dim) * 0.3
            no_improve = 0

            while calls < self.budget and time.time() < deadline and no_improve < 15:
                dim_idx = rng.integers(self.dim)
                sign = rng.choice([-1.0, 1.0])
                y = x.copy()
                y[dim_idx] = np.clip(y[dim_idx] + sign * steps[dim_idx], -5.0, 5.0)
                fy = func(y)
                count_call()

                if fy < fx:
                    x = y
                    fx = fy
                    steps[dim_idx] *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    no_improve = 0
                else:
                    steps[dim_idx] *= 0.8
                    if steps[dim_idx] < 0.005:
                        steps[dim_idx] = 0.005
                    no_improve += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
