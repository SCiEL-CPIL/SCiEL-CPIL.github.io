import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-restart Powell's method optimizer for BBOB-style f19 (Griewank-Rosenbrock).
    Powell's method is derivative-free and handles the Rosenbrock valley structure
    well by rotating search directions to align with the valley floor.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-restart Powell
        # Powell is derivative-free and efficient for low-dimensional non-convex problems.
        # We use multiple restarts to escape local minima of the Griewank component.
        
        # Reserve some budget for a final high-precision polish
        polish_budget = 50
        search_budget = budget - polish_budget
        
        num_restarts = 12
        # Estimate calls per Powell run. Powell uses ~N+1 evals per cycle.
        # With N=5, that's ~6 evals per "step" in the direction update logic, 
        # but effectively more due to line searches. 
        # Let's cap maxiter to ensure we don't overshoot.
        maxiter_per_restart = max(20, search_budget // num_restarts)

        # Initial diverse starting points
        # 1. Zero
        # 2. Ones
        # 3-12. Random
        starts = [np.zeros(n), np.ones(n)]
        for _ in range(num_restarts - 2):
            starts.append(rng.uniform(-4.5, 4.5, size=n))

        for x_start in starts:
            if calls >= search_budget or time.time() >= deadline:
                break
            
            x_start = np.clip(x_start, -5.0, 5.0)
            
            try:
                # Powell's method
                # xatol and fatol control convergence
                result = minimize(
                    lambda x: evaluate(x),
                    x_start,
                    method='Powell',
                    options={
                        'maxiter': maxiter_per_restart,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'disp': False
                    }
                )
            except Exception:
                pass

        # Phase 2: Polish
        # Use Nelder-Mead or L-BFGS-B for final precision if budget remains.
        # L-BFGS-B is very fast for final convergence if the starting point is good.
        if calls < budget and time.time() < deadline and best_x is not None:
            try:
                remaining = budget - calls
                # L-BFGS-B for high precision
                result = minimize(
                    lambda x: evaluate(x),
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': max(10, remaining),
                        'ftol': 1e-15,
                        'gtol': 1e-12,
                    }
                )
            except Exception:
                pass

        # Phase 3: If still budget left and time, try one more NM polish
        if calls < budget and time.time() < deadline and best_x is not None:
            try:
                remaining = budget - calls
                # Small simplex for NM
                simplex = np.array([best_x])
                for i in range(n):
                    p = best_x.copy()
                    p[i] += 0.01
                    simplex = np.vstack([simplex, p])
                
                result = minimize(
                    lambda x: evaluate(x),
                    best_x,
                    method='Nelder-Mead',
                    initial_simplex=simplex,
                    options={
                        'maxiter': max(10, remaining),
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                    }
                )
            except Exception:
                pass

        return best_x, best_f
