"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with BFGS refinement and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, keep 20s margin
        rng = np.random.default_rng(12345)
        n = self.dim
        budget = self.budget

        # We'll track all evaluations
        eval_count = [0]

        def wrapped_func(x):
            if eval_count[0] >= budget:
                return 1e10
            x_clipped = np.clip(x, -5.0, 5.0)
            val = float(func(x_clipped))
            eval_count[0] += 1
            return val

        best_x = None
        best_f = np.inf

        # Strategy: multiple restarts with different initial points
        # Use a mix of random and structured initial points
        
        # First, do a quick random search to get some good starting points
        n_random = min(20, budget // 10)
        random_points = []
        for i in range(n_random):
            x = rng.uniform(-5, 5, n)
            fx = wrapped_func(x)
            random_points.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort random points by fitness
        random_points.sort(key=lambda t: t[0])
        
        # Take top few as seeds for local optimization
        n_seeds = min(8, len(random_points))
        seeds = [random_points[i][1] for i in range(n_seeds)]
        
        # Also add origin as a seed
        if not any(np.allclose(s, 0.0) for s in seeds):
            seeds.append(np.zeros(n))
        
        # For each seed, run Nelder-Mead then BFGS refinement
        for seed in seeds:
            if eval_count[0] >= budget:
                break
            if time.time() > deadline - 30:
                break
                
            # Nelder-Mead phase
            try:
                res_nm = minimize(
                    wrapped_func,
                    seed,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(200, budget - eval_count[0]),
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x.copy()
            except Exception:
                pass

            if eval_count[0] >= budget:
                break
            if time.time() > deadline - 15:
                break

            # BFGS refinement from Nelder-Mead result
            try:
                res_bfgs = minimize(
                    wrapped_func,
                    res_nm.x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': min(50, budget - eval_count[0]),
                        'gtol': 1e-6,
                        'ftol': 1e-10
                    }
                )
                if res_bfgs.fun < best_f:
                    best_f = res_bfgs.fun
                    best_x = res_bfgs.x.copy()
            except Exception:
                pass

        # If we still have budget, do more random restarts
        remaining = budget - eval_count[0]
        if remaining > 50 and time.time() < deadline - 20:
            n_more = min(remaining // 10, 10)
            for i in range(n_more):
                if eval_count[0] >= budget:
                    break
                if time.time() > deadline - 10:
                    break
                x = rng.uniform(-5, 5, n)
                try:
                    res = minimize(
                        wrapped_func,
                        x,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * n,
                        options={
                            'maxiter': min(30, budget - eval_count[0]),
                            'gtol': 1e-5,
                            'ftol': 1e-8
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Final check: if best_x is None, use a default
        if best_x is None:
            best_x = np.zeros(n)
            best_f = wrapped_func(best_x)

        return np.clip(best_x, -5.0, 5.0), best_f
# EVOLVE-BLOCK-END
