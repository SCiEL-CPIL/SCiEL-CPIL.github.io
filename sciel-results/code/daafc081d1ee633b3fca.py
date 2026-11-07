"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


class Optimizer:
    """Hybrid optimizer: differential evolution for global search + Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls_made = 0
        self._start_time = 0.0

    def _evaluate(self, func, x):
        """Evaluate function with call counting and deadline guard."""
        if self._calls_made >= self.budget:
            return np.inf
        if time.time() - self._start_time > 320:
            return np.inf
        self._calls_made += 1
        try:
            return float(func(x))
        except Exception:
            return np.inf

    def __call__(self, func):
        self._start_time = time.time()
        self._calls_made = 0
        dim = self.dim
        budget = self.budget

        # Wrap func to count calls through scipy
        def counted_func(x):
            return self._evaluate(func, x)

        # Phase 1: Differential evolution for global exploration
        # Use a moderate budget for DE - it's good at finding the basin of attraction
        de_budget = min(300, budget)
        
        # Set up DE with reasonable parameters for 5D sphere
        # popsize=15 gives 75 individuals, each generation uses 75 calls
        # With 300 calls, we get about 4 generations
        try:
            de_result = differential_evolution(
                counted_func,
                bounds=[(-5, 5)] * dim,
                maxiter=100,
                popsize=10,
                tol=1e-8,
                seed=42,
                init='latinhypercube',
                polish=False,
                updating='deferred'
            )
        except Exception:
            de_result = None

        # Get the best point from DE
        if de_result is not None and de_result.success:
            best_x = de_result.x
            best_fx = de_result.fun
        else:
            # Fallback: try a few random points
            rng = np.random.default_rng(42)
            best_x = None
            best_fx = np.inf
            for _ in range(min(20, budget - self._calls_made)):
                x = rng.uniform(-5, 5, dim)
                fx = self._evaluate(func, x)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x
            if best_x is None:
                best_x = np.zeros(dim)
                best_fx = self._evaluate(func, best_x)

        # Phase 2: Nelder-Mead refinement from the best point
        remaining = budget - self._calls_made
        if remaining > 10:
            try:
                nm_result = minimize(
                    counted_func,
                    x0=best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if nm_result.fun < best_fx:
                    best_x = nm_result.x
                    best_fx = nm_result.fun
            except Exception:
                pass

        # Phase 3: Final fine-tuning with small steps if budget remains
        remaining = budget - self._calls_made
        if remaining > 5 and best_x is not None:
            rng = np.random.default_rng(123)
            sigma = 0.01
            for _ in range(remaining):
                if time.time() - self._start_time > 325:
                    break
                x_new = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fx_new = self._evaluate(func, x_new)
                if fx_new < best_fx:
                    best_x = x_new
                    best_fx = fx_new
                    sigma *= 1.2
                else:
                    sigma *= 0.9
                sigma = max(sigma, 1e-10)

        return best_x, best_fx
# EVOLVE-BLOCK-END
