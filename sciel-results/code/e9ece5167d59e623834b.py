"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with restarts and a deadline guard for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        deadline = time.time() + 280.0  # safety margin before 300s limit

        # Count calls to ensure we don't exceed budget
        call_count = [0]
        max_calls = self.budget

        def safe_call(x):
            if call_count[0] >= max_calls:
                return None
            call_count[0] += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            if not np.isfinite(val):
                val = 1e10
            return val

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Run multiple restarts of (1+1)-ES
        # Each restart gets a fraction of the budget
        # We use a portfolio approach: several short runs, then one long refinement

        # Strategy:
        # - Phase 1: Several quick exploratory runs (5 restarts, ~50 evals each)
        # - Phase 2: One long refinement from the best point (~300 evals)
        # - Phase 3: If time/budget remains, a final fine-tuning phase

        rng = np.random.default_rng(42)

        # Phase 1: Exploratory restarts
        n_exploratory = 6
        budget_per_expl = min(50, max(20, (max_calls - 100) // n_exploratory))

        for restart in range(n_exploratory):
            if call_count[0] >= max_calls:
                break
            if time.time() > deadline:
                break

            # Start from a random point, biased towards center for later restarts
            if restart == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                # Mix: some random, some near best
                if best_x is not None and restart % 2 == 0:
                    x = best_x + rng.normal(0, 1.0, dim)
                else:
                    x = rng.uniform(-5, 5, dim)

            x = np.clip(x, -5, 5)
            fx = safe_call(x)
            if fx is None:
                break
            update_best(x, fx)

            sigma = 2.0
            success_count = 0
            total_trials = 0
            consecutive_failures = 0

            for _ in range(budget_per_expl - 1):
                if call_count[0] >= max_calls:
                    break
                if time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break

                total_trials += 1
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    success_count += 1
                    consecutive_failures = 0
                    # Increase sigma on success
                    sigma *= 1.2
                else:
                    consecutive_failures += 1
                    # Decrease sigma on failure
                    sigma *= 0.9

                # Adaptive step size: 1/5 rule approximation
                if total_trials >= 10:
                    rate = success_count / total_trials
                    if rate > 0.3:
                        sigma *= 1.1
                    elif rate < 0.1:
                        sigma *= 0.85
                    # Reset counters for next window
                    success_count = 0
                    total_trials = 0

                # Early stop if stuck
                if consecutive_failures > 15 and sigma < 0.01:
                    break

        # Phase 2: Long refinement from best point
        if call_count[0] < max_calls and best_x is not None:
            if time.time() > deadline:
                break

            x = best_x.copy()
            fx = best_f
            sigma = 1.0  # Start with moderate step size for refinement

            success_count = 0
            total_trials = 0
            consecutive_failures = 0
            stagnation_counter = 0

            while call_count[0] < max_calls:
                if time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break

                total_trials += 1
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    success_count += 1
                    consecutive_failures = 0
                    stagnation_counter = 0
                    sigma *= 1.15
                else:
                    consecutive_failures += 1
                    stagnation_counter += 1
                    sigma *= 0.92

                # Adaptive step size adjustment
                if total_trials >= 10:
                    rate = success_count / total_trials
                    if rate > 0.3:
                        sigma *= 1.1
                    elif rate < 0.1:
                        sigma *= 0.85
                    success_count = 0
                    total_trials = 0

                # If very stuck, try a larger jump
                if consecutive_failures > 20:
                    sigma = max(sigma * 2.0, 0.5)
                    consecutive_failures = 0

                # Stop if sigma is too small
                if sigma < 1e-6:
                    break

        # Phase 3: Final fine-tuning if budget remains
        remaining = max_calls - call_count[0]
        if remaining > 10 and best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1  # Small steps for fine-tuning

            for _ in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                if call_count[0] >= max_calls:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 0.9  # Slightly reduce step
                else:
                    sigma *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
