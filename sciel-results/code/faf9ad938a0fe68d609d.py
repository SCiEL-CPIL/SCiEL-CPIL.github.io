"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DFTR-style optimizer with finite-difference gradient and adaptive trust region."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s kill

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Wrap func to track calls and enforce deadline
        call_count = [0]

        def safe_func(x):
            if time.time() > deadline:
                raise TimeoutError
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = float(func(x))
            call_count[0] += 1
            if call_count[0] > budget:
                raise BudgetExceeded
            return val

        class BudgetExceeded(Exception):
            pass

        # Initial point: center of box (good for symmetric functions)
        x = np.zeros(dim)
        try:
            fx = safe_func(x)
        except (TimeoutError, BudgetExceeded):
            return x, fx if 'fx' in dir() else 0.0

        best_x, best_fx = x.copy(), fx

        # Finite-difference gradient
        def gradient(x, fx, eps=1e-5):
            g = np.zeros(dim)
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] += eps
                x_plus = np.clip(x_plus, -5, 5)
                try:
                    f_plus = safe_func(x_plus)
                except (TimeoutError, BudgetExceeded):
                    return None
                x_minus = x.copy()
                x_minus[i] -= eps
                x_minus = np.clip(x_minus, -5, 5)
                try:
                    f_minus = safe_func(x_minus)
                except (TimeoutError, BudgetExceeded):
                    return None
                # Handle boundary: use one-sided if clipped
                if x[i] + eps > 5 or x[i] + eps < -5:
                    g[i] = (f_plus - fx) / (x_plus[i] - x[i])
                elif x[i] - eps > 5 or x[i] - eps < -5:
                    g[i] = (fx - f_minus) / (x[i] - x_minus[i])
                else:
                    g[i] = (f_plus - f_minus) / (2 * eps)
            return g

        # Adaptive trust-region radius
        delta = 2.0  # initial trust region radius
        min_delta = 1e-8
        max_delta = 8.0

        for _ in range(budget - 1):
            if time.time() > deadline:
                break

            # Compute gradient
            try:
                g = gradient(x, fx)
            except (TimeoutError, BudgetExceeded):
                break

            if g is None:
                break

            # If gradient is tiny, we're near a minimum
            if np.linalg.norm(g) < 1e-10:
                break

            # Compute steepest descent step
            grad_norm = np.linalg.norm(g)
            if grad_norm < 1e-15:
                break
            d = -g / grad_norm

            # Try step at trust region boundary
            step = delta * d
            x_trial = np.clip(x + step, -5, 5)

            try:
                f_trial = safe_func(x_trial)
            except (TimeoutError, BudgetExceeded):
                break

            # Check improvement
            if f_trial < fx:
                # Successful step: move and expand trust region
                x, fx = x_trial, f_trial
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                delta = min(2.0 * delta, max_delta)
            else:
                # Unsuccessful: shrink trust region
                delta *= 0.5
                if delta < min_delta:
                    # Try a small random perturbation to escape
                    x = np.clip(x + 0.1 * rng.standard_normal(dim), -5, 5)
                    try:
                        fx = safe_func(x)
                    except (TimeoutError, BudgetExceeded):
                        break
                    delta = 0.5

        return best_x, best_fx
# EVOLVE-BLOCK-END
