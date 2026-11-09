"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with 1/5 success rule, time-bounded, plus Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx
        sigma = 2.0
        calls = 1
        stuck_count = 0
        last_improvement_fx = fx

        # Reserve last 30% of budget for polish
        es_budget = int(0.7 * (self.budget - 1))

        for i in range(es_budget):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                if fx < last_improvement_fx - 1e-10:
                    last_improvement_fx = fx
                    stuck_count = 0
                else:
                    stuck_count += 1
                # Adaptive sigma: increase on success, but limit growth
                sigma = min(sigma * 1.3, 5.0)
            else:
                sigma *= 0.85
                stuck_count += 1

            # If stuck for many iterations, jump to explore new region
            if stuck_count > 30 and i < es_budget * 0.8:
                sigma = max(sigma, 3.0)
                stuck_count = 0

            if fx < best_fx:
                best_x, best_fx = x.copy(), fx

        # Nelder-Mead polish pass on the best point found
        if time.time() < deadline - 5.0:
            try:
                nm_budget = max(1, int(0.5 * (self.budget - calls)))
                result = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 150,
                        'maxfev': nm_budget,
                        'xatol': 1e-8,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if result.fun < best_fx:
                    best_x = result.x
                    best_fx = result.fun
            except Exception:
                pass

        # Final high-precision polish with L-BFGS-B if time remains
        if time.time() < deadline - 2.0 and calls < self.budget:
            try:
                remaining = self.budget - calls
                result2 = minimize(
                    func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * self.dim,
                    options={
                        'maxiter': 50,
                        'maxfev': remaining,
                        'ftol': 1e-15,
                        'gtol': 1e-12,
                        'eps': 1e-8
                    }
                )
                if result2.fun < best_fx:
                    best_x = result2.x
                    best_fx = result2.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
