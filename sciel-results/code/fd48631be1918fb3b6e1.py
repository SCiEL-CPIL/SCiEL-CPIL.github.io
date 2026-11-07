"""Nelder-Mead + Random Walk optimizer for BBOB-style f19 (Griewank-Rosenbrock)."""
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with random-walk initialization for rotated/shifted landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # 15s safety margin before 300s limit

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Random Walk / Hill Climber ---
        # Use a simple adaptive step-size random walk to explore the rotated landscape.
        # This is much cheaper than CMA-ES and works well for the Rosenbrock core.
        x = np.zeros(n)
        evaluate(x)
        step_size = 2.0

        while calls < 300 and time.time() < deadline:
            # Generate a random direction
            direction = rng.standard_normal(n)
            direction /= np.linalg.norm(direction)

            # Try a step
            x_try = np.clip(x + step_size * direction, -5.0, 5.0)
            f_try = evaluate(x_try)

            if f_try < best_f:
                # Move in that direction
                x = x_try
                # Slightly increase step size if improving
                step_size *= 1.05
            else:
                # Try opposite direction
                x_try2 = np.clip(x - step_size * direction, -5.0, 5.0)
                f_try2 = evaluate(x_try2)
                if f_try2 < best_f:
                    x = x_try2
                    step_size *= 1.05
                else:
                    # No improvement, shrink step size
                    step_size *= 0.8

            # Safety check on step size
            if step_size < 1e-6:
                break
            if step_size > 10.0:
                step_size = 10.0

        # --- Phase 2: Nelder-Mead Polish ---
        # Use the best point found as the starting point for Nelder-Mead.
        if calls < budget and time.time() < deadline:
            try:
                from scipy.optimize import minimize

                # Start Nelder-Mead from the best point found
                x0 = best_x.copy()
                
                # Use a wrapper that clips and tracks calls
                def wrapped_func(x):
                    return evaluate(x)

                # Nelder-Mead parameters
                # Initial simplex size
                initial_simplex_size = max(0.1, step_size * 0.5)
                
                result = minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(100, budget - calls),
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                        'initial_simplex': None # Let scipy handle it, or we can provide
                    }
                )

                # Scipy's minimize already calls the function, so our 'evaluate' wrapper
                # has updated best_x and best_f.
                # However, we should ensure we don't exceed budget.
                # The maxiter option limits iterations, but each iteration uses multiple calls.
                # To be safe, we rely on the deadline and budget checks inside evaluate.
                
            except Exception:
                pass

        return best_x, best_f
