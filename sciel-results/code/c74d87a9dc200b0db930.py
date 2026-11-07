"""Black-box continuous optimizer on BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """
    Robust optimizer for BBOB f11 (Discus):
    - Phase 1: Sobol sampling for initial exploration.
    - Phase 2: Multi-start L-BFGS-B local refinement from top candidates.
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

        # Phase 1: Sobol sampling to build initial diverse set
        n_initial = min(64, self.budget // 5)
        # Use Sobol sequence for better coverage than random
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        # Generate points in [0, 1]
        sobol_points = sampler.random(n_initial)
        # Scale to [-5, 5]
        X_initial = sobol_points * 10.0 - 5.0
        
        evaluations = []
        for x in X_initial:
            if calls_used >= self.budget - 20:
                break
            fx = safe_call(x)
            evaluations.append((fx, x))

        if not evaluations:
            # Fallback to random if Sobol fails or budget is tiny
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_call(x)
            evaluations.append((fx, x))

        # Sort by function value
        evaluations.sort(key=lambda t: t[0])
        
        # Phase 2: Local refinement from the best few points
        # We have used ~n_initial calls. Remaining budget for local search.
        remaining_budget = self.budget - calls_used
        
        # Select top candidates for local refinement
        n_restarts = min(5, len(evaluations))
        # Each restart gets a portion of the remaining budget
        # L-BFGS-B is efficient, so we can afford more iterations
        max_iter_per_restart = max(10, remaining_budget // (n_restarts + 1))
        
        bounds = [(-5, 5)] * self.dim
        
        for i in range(n_restarts):
            if time.time() > deadline - 10:
                break
            if calls_used >= self.budget - 5:
                break
                
            fx0, x0 = evaluations[i]
            
            # Use the best current point as the starting point for the first restart
            # and the other good points for subsequent restarts
            # Actually, just use the specific point from the sorted list
            
            def lbfgs_func(x):
                return safe_call(x)

            try:
                res = minimize(
                    lbfgs_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': max_iter_per_restart,
                        'gtol': 1e-10,
                        'ftol': 1e-12,
                    }
                )
                # Update global best if this run found something better
                # safe_call already updates best_x and best_f, but res.fun might be
                # slightly different due to numerical precision or if it stopped early.
                # We trust safe_call's update.
            except Exception:
                pass

        # Phase 3: If budget remains, try a few more random restarts with local search
        if calls_used < self.budget - 20 and time.time() < deadline - 30:
            # Estimate how many more restarts we can afford
            # Assume ~20 calls per restart on average
            possible_restarts = (self.budget - calls_used) // 30
            for _ in range(possible_restarts):
                if time.time() > deadline - 10:
                    break
                if calls_used >= self.budget - 10:
                    break
                
                x0 = rng.uniform(-5, 5, self.dim)
                fx0 = safe_call(x0)
                
                if calls_used < self.budget - 5:
                    def lbfgs_func2(x):
                        return safe_call(x)
                    try:
                        res = minimize(
                            lbfgs_func2,
                            x0,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={
                                'maxiter': 20,
                                'gtol': 1e-8,
                            }
                        )
                    except Exception:
                        pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            if calls_used < self.budget:
                best_f = safe_call(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
