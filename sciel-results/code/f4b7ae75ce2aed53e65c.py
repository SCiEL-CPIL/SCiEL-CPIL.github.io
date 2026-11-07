"""Black-box continuous optimizer on BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Hybrid optimizer for BBOB f11 (Discus):
    - Phase 1: (1+1)-ES for rapid global exploration and basin finding.
    - Phase 2: Nelder-Mead local refinement from the best point found.
    - Deadline guard to ensure termination within time limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_used = 0

        def check_budget():
            nonlocal calls_used
            if calls_used >= self.budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")

        def safe_call(x):
            nonlocal calls_used, best_f, best_x
            check_budget()
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: (1+1)-ES exploration
        # Use ~60% of budget for global search via (1+1)-ES
        es_budget = int(self.budget * 0.6)
        x = rng.uniform(-5, 5, self.dim)
        fx = safe_call(x)
        sigma = 2.0
        success_count = 0
        window = 5

        for i in range(es_budget - 1):
            if time.time() > deadline - 20:
                break
            if calls_used >= self.budget:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = safe_call(y)

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
                success_count += 1
            else:
                sigma *= 0.85

            # Adapt sigma based on success rate
            if (i + 1) % window == 0:
                if success_count / window > 0.5:
                    sigma *= 1.2
                elif success_count / window < 0.2:
                    sigma *= 0.7
                success_count = 0

        # Phase 2: Local refinement using Nelder-Mead
        # Use remaining budget for local refinement
        remaining = self.budget - calls_used
        if remaining > 10 and best_x is not None:
            # Run Nelder-Mead from the best point found
            nm_budget = min(remaining, 200)  # Cap NM budget to ensure it finishes

            def nm_func(x):
                return safe_call(x)

            try:
                res = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': nm_budget,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True,
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        # Phase 3: If budget remains, do a few more random restarts + quick NM
        if calls_used < self.budget and time.time() < deadline - 30:
            n_restarts = min(5, (self.budget - calls_used) // 40)
            for _ in range(n_restarts):
                if calls_used >= self.budget - 20:
                    break
                if time.time() > deadline - 10:
                    break

                x0 = rng.uniform(-5, 5, self.dim)
                fx0 = safe_call(x0)

                if calls_used < self.budget - 10:
                    def nm_func2(x):
                        return safe_call(x)

                    try:
                        res = minimize(
                            nm_func2,
                            x0,
                            method='Nelder-Mead',
                            options={
                                'maxiter': min(20, self.budget - calls_used - 5),
                                'xatol': 1e-6,
                                'fatol': 1e-8,
                                'adaptive': True,
                            }
                        )
                    except Exception:
                        pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = safe_call(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
