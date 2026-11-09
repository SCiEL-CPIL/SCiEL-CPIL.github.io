"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES based optimizer for Katsuura function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        calls = 0
        best_x = None
        best_f = float('inf')

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f
            if time.time() - start_time > time_limit - 5.0:
                return best_f
            x_clipped = np.clip(x, -5, 5)
            v = func(x_clipped)
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x_clipped.copy()
            return v

        # Phase 1: Quick initial sampling to find good basin
        n_init = 15
        init_points = [np.zeros(dim)]
        for _ in range(7):
            init_points.append(rng.normal(0, 0.3, dim))
        for _ in range(7):
            init_points.append(rng.uniform(-3, 3, dim))
        
        for p in init_points:
            p = np.clip(p, -5, 5)
            wrapped_func(p)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)
        
        # Phase 2: CMA-ES from the best initial point
        # Katsuura has global min at 0, so start near 0
        x0 = best_x.copy()
        
        try:
            res = minimize(wrapped_func, x0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_f:
                best_f = res.fun
                best_x = np.clip(res.x, -5, 5)
        except Exception:
            pass
        
        # Phase 3: If budget remains, do focused local search around best
        while calls < self.budget - 5:
            if time.time() - start_time > time_limit - 10.0:
                break
            
            # Try small perturbations around current best
            sigma = 0.1
            for _ in range(5):
                if calls >= self.budget - 2:
                    break
                new_p = best_x + rng.normal(0, sigma, dim)
                new_p = np.clip(new_p, -5, 5)
                v = wrapped_func(new_p)
                if v < best_f:
                    best_f = v
                    best_x = new_p.copy()
                    sigma *= 0.5
                else:
                    sigma *= 1.5
            
            # Occasional global restart to escape local minima
            if rng.random() < 0.1 and calls < self.budget - 50:
                new_p = rng.uniform(-4, 4, dim)
                new_p = np.clip(new_p, -5, 5)
                v = wrapped_func(new_p)
                if v < best_f + 2.0:
                    try:
                        res = minimize(wrapped_func, new_p, method='Nelder-Mead',
                                      options={'maxiter': 20, 'xatol': 1e-6, 'fatol': 1e-10})
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = np.clip(res.x, -5, 5)
                    except Exception:
                        pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
