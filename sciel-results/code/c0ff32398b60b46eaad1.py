"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        best_x = x.copy()
        best_fx = fx

        # Track step size adaptation
        successes = 0
        total_steps = 0

        for i in range(self.budget - 1):
            # Check time limit
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                successes += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                sigma *= 0.85

            total_steps += 1

        # If we have time left, do a local refinement phase
        if time.time() < deadline - 10:
            # Use Nelder-Mead for local refinement
            try:
                from scipy.optimize import minimize

                # Run several Nelder-Mead restarts from different points
                nm_results = []
                nm_starts = [best_x.copy()]

                # Add a few random restarts
                for _ in range(5):
                    nm_starts.append(rng.uniform(-5, 5, self.dim))

                for start in nm_starts:
                    if time.time() > deadline - 5:
                        break
                    try:
                        result = minimize(
                            func, start, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                        )
                        if result.fun < best_fx:
                            best_fx = result.fun
                            best_x = result.x.copy()
                    except Exception:
                        pass

            except ImportError:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
