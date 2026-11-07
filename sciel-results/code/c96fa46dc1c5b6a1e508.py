"""Optimizer for BBOB-style Rosenbrock (f8) with hidden transformations."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize as scipy_minimize


class Optimizer:
    """Hybrid optimizer: Multi-start Nelder-Mead global search followed by L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        # Wrap func to count calls and track best
        state = {'calls': 0, 'best_x': None, 'best_f': np.inf}

        def wrapped_func(x):
            state['calls'] += 1
            if state['calls'] > self.budget:
                raise RuntimeError("Budget exceeded")
            fx = func(x)
            if fx < state['best_f']:
                state['best_f'] = fx
                state['best_x'] = x.copy()
            return fx

        rng = np.random.default_rng(42)

        best_overall_x = None
        best_overall_f = np.inf

        # Phase 1: Global search with multiple Nelder-Mead restarts
        # Nelder-Mead is derivative-free and robust for finding the valley basin
        num_restarts = 10
        budget_per_restart = 30  # Limit calls per global search to save budget for polish

        for restart in range(num_restarts):
            if state['calls'] >= self.budget:
                break
            if time.time() > deadline - 30:
                break

            # Diverse initialization
            if restart == 0:
                x0 = np.zeros(self.dim)
            elif restart == 1:
                x0 = np.ones(self.dim) * 0.5
            elif restart == 2:
                x0 = np.ones(self.dim) * -0.5
            else:
                x0 = rng.uniform(-4, 4, self.dim)

            try:
                result = scipy_minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100, 
                        'maxfev': budget_per_restart,
                        'xatol': 1e-6, 
                        'fatol': 1e-6
                    }
                )
                if result.fun < best_overall_f:
                    best_overall_f = result.fun
                    best_overall_x = result.x.copy()
            except Exception:
                pass

        # Phase 2: Polish with L-BFGS-B
        # L-BFGS-B is very efficient for smooth functions like Rosenbrock once in the basin
        if best_overall_x is not None and state['calls'] < self.budget:
            remaining_budget = self.budget - state['calls']
            if remaining_budget > 10 and time.time() < deadline - 10:
                try:
                    result = scipy_minimize(
                        wrapped_func,
                        best_overall_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={
                            'maxiter': 500, 
                            'maxfun': remaining_budget,
                            'ftol': 1e-15, 
                            'gtol': 1e-15
                        }
                    )
                    if result.fun < best_overall_f:
                        best_overall_f = result.fun
                        best_overall_x = result.x.copy()
                except Exception:
                    pass

        # Phase 3: Final fine-tuning if budget remains (optional, usually L-BFGS-B is enough)
        # If L-BFGS-B didn't converge fully, a small Nelder-Mead can help, but it's risky.
        # We'll skip this to avoid noise, L-BFGS-B is strong.

        if best_overall_x is None:
            best_overall_x = np.zeros(self.dim)

        return best_overall_x, best_overall_f
# EVOLVE-BLOCK-END
