"""Rewritten program for BBOB-style Rastrigin optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Rastrigin-aware hybrid: Multi-start CMA-ES with adaptive population + Local Polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit

        rng = np.random.default_rng(42)

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial Exploration
        n_init = 15
        for i in range(n_init):
            if time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, dim)
            counted_func(x)

        # Phase 2: Multi-start CMA-ES
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and time.time() < deadline:
            # Use 4 runs with decreasing budget allocation
            # First run gets the most budget for thorough exploration
            run_budgets = [
                int(remaining_budget * 0.35),
                int(remaining_budget * 0.25),
                int(remaining_budget * 0.20),
                int(remaining_budget * 0.15)
            ]

            current_best_x = best_x.copy()

            for run_idx, rb in enumerate(run_budgets):
                if calls_made >= budget or time.time() > deadline - 5:
                    break

                if rb < 10:
                    continue

                # Determine starting point
                if run_idx == 0:
                    x_start = current_best_x.copy()
                else:
                    # Perturb the current best with decreasing step size
                    sigma = 0.8 / (run_idx + 1)
                    x_start = current_best_x + rng.normal(0, sigma, dim)
                    x_start = np.clip(x_start, -5, 5)

                # Adaptive population size: larger for earlier runs
                popsize = max(20, 35 - 5 * run_idx)

                try:
                    res = minimize(
                        counted_func,
                        x_start,
                        method='CMA-ES',
                        options={
                            'maxiter': 300,
                            'maxfev': rb,
                            'tol': 1e-12,
                            'seed': 100 + run_idx * 7,
                            'popsize': popsize,
                            'beta1': 0.5,
                            'beta2': 0.5,
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    # Fallback to Powell if CMA-ES fails
                    try:
                        res = minimize(
                            counted_func,
                            x_start,
                            method='Powell',
                            options={'maxiter': 100, 'maxfev': rb, 'ftol': 1e-12}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass

        # Phase 3: Final Polish with Nelder-Mead
        if calls_made < budget - 5 and time.time() < deadline - 5:
            try:
                res = minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 4: Additional local refinement with multiple restarts
        if calls_made < budget - 10 and time.time() < deadline - 10:
            # Try a few small perturbations around the best point
            for _ in range(3):
                if calls_made >= budget - 5 or time.time() > deadline - 5:
                    break
                x_perturb = best_x + rng.normal(0, 0.1, dim)
                x_perturb = np.clip(x_perturb, -5, 5)
                try:
                    res = minimize(
                        counted_func,
                        x_perturb,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 50,
                            'xatol': 1e-12,
                            'fatol': 1e-14,
                            'adaptive': True
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
