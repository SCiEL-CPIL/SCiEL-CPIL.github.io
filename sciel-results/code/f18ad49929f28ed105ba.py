"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """BBOB-style optimizer using a combination of CMA-ES-like and Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self.calls = 0

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self.calls = 0

        def wrapper(x):
            self.calls += 1
            return func(x)

        # Start with a few random evaluations to find a good starting point
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # Try a few random starting points
        n_init = 5
        for _ in range(n_init):
            x0 = rng.uniform(-5, 5, dim)
            f0 = wrapper(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Use Nelder-Mead from the best starting point
        # Nelder-Mead is derivative-free and works well for low-dimensional problems
        try:
            result = minimize(
                wrapper,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'maxfev': budget // 3,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x
        except Exception:
            pass

        # If we still have budget, do a local search with small perturbations
        if self.calls < budget:
            sigma = 0.1
            for _ in range(budget - self.calls):
                if self.calls >= budget:
                    break
                y = best_x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = wrapper(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
