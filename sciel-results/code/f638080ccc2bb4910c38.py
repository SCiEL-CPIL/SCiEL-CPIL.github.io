"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Adaptive 1+1 Evolution Strategy with multi-start restarts.
    This is a fundamentally different design family from CMA-ES:
    - No population, no covariance matrix
    - Single trajectory with adaptive step size
    - Multiple restarts to escape local optima
    - Very fast per-iteration, allows many more evaluations
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, 15s safety margin
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        # Phase 1: Multi-start with short 1+1 ES runs
        # Allocate budget across multiple restarts
        n_restarts = 10
        budget_per_restart = self.budget // n_restarts

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if calls_made >= self.budget:
                break

            # Initialize starting point
            if r == 0:
                x = rng.uniform(-5, 5, self.dim)
            else:
                # For subsequent restarts, try different strategies
                if r % 3 == 1:
                    # Random point
                    x = rng.uniform(-5, 5, self.dim)
                elif r % 3 == 2:
                    # Perturb best known point
                    if best_x is not None:
                        x = best_x + rng.normal(0, 1.5, self.dim)
                        x = np.clip(x, -5, 5)
                    else:
                        x = rng.uniform(-5, 5, self.dim)
                else:
                    # Point near center with some noise
                    x = rng.normal(0, 1.0, self.dim)
                    x = np.clip(x, -5, 5)

            fx = func(x)
            calls_made += 1

            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            # Run 1+1 ES from this starting point
            sigma = 1.0
            success_count = 0
            failure_count = 0
            window = 10  # Adaptation window

            while calls_made < self.budget:
                if time.time() > deadline:
                    break

                # Generate candidate
                direction = rng.standard_normal(self.dim)
                candidate = x + sigma * direction
                candidate = np.clip(candidate, -5, 5)

                f_candidate = func(candidate)
                calls_made += 1

                if f_candidate < fx:
                    # Success
                    x = candidate
                    fx = f_candidate
                    success_count += 1

                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    # Failure
                    failure_count += 1

                # Adapt step size every 'window' steps
                if success_count + failure_count >= window:
                    # 1/5 success rule: increase sigma if >20% success, decrease otherwise
                    if success_count > 0.2 * window:
                        sigma *= 1.5
                    elif success_count < 0.2 * window:
                        sigma *= 0.7
                    else:
                        sigma *= 1.0

                    sigma = np.clip(sigma, 1e-8, 10.0)
                    success_count = 0
                    failure_count = 0

                # Early termination if sigma is very small
                if sigma < 1e-6:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
