"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid (1+1)-ES + local polish for BBOB f16 Weierstrass."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        rng = np.random.default_rng(42)

        call_count = 0

        def count_func(x):
            nonlocal call_count
            if call_count >= self.budget:
                return np.inf
            call_count += 1
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (self.dim,):
                x = np.resize(x, self.dim)
            x = np.clip(x, -5.0, 5.0)
            val = float(func(x))
            if not np.isfinite(val):
                return 1e10
            return val

        # Phase 1: (1+1)-ES with adaptive sigma
        x = rng.uniform(-4.0, 4.0, self.dim)
        fx = count_func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 1.5
        no_improve = 0
        max_no_improve = 30

        phase1_budget = int(0.8 * self.budget)  # 400 calls
        for i in range(phase1_budget - 1):
            if time.time() > deadline:
                break
            if call_count >= self.budget:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5.0, 5.0)
            fy = count_func(y)

            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                    no_improve = 0
                sigma *= 1.3
            else:
                sigma *= 0.85
                no_improve += 1
                if no_improve > max_no_improve:
                    # Restart from a random point near best
                    x = np.clip(best_x + rng.normal(0, sigma * 0.5, self.dim), -5.0, 5.0)
                    fx = count_func(x)
                    no_improve = 0
                    sigma = max(sigma, 0.5)

        # Phase 2: Local polish with scipy
        if call_count < self.budget and time.time() < deadline:
            try:
                res = minimize(
                    count_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True,
                    }
                )
                if res.fun < best_fx:
                    best_x = np.clip(res.x, -5.0, 5.0)
                    best_fx = res.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
