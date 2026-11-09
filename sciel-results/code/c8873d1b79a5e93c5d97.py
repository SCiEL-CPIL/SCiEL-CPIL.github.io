import numpy as np
import time

class Optimizer:
    """
    Hybrid optimizer for BBOB-style Sphere f1.
    Strategy:
    1. Structured initial exploration to find a good basin quickly (early progress).
    2. CMA-ES for robust global-to-local convergence, handling rotation/shift.
    3. L-BFGS-B for high-precision polishing.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(n)

        def eval_point(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # --- Phase 1: Structured Exploration ---
        # We want to find a good starting point for CMA-ES quickly.
        # The minimum is in [-4, 4]^5.
        # Evaluate origin.
        f0 = eval_point(np.zeros(n))
        if f0 is None:
            return best_x, best_f

        # Evaluate axis points at several scales to estimate gradient direction roughly
        # and to provide diverse points for the initial "best_x" selection if origin is bad.
        # This helps if the shift is significant.
        scales = [4.0, 2.0, 1.0, 0.5]
        # We will keep the best point found in this phase as x0 for CMA-ES.
        
        # Note: We are careful not to spend too much budget here.
        # 1 (origin) + 5 dims * 2 signs * 4 scales = 40 calls.
        # This is 8% of budget. Acceptable.
        
        for scale in scales:
            if calls >= budget or time.time() > deadline:
                break
            for i in range(n):
                if calls >= budget or time.time() > deadline:
                    break
                x_plus = np.zeros(n)
                x_plus[i] = scale
                eval_point(x_plus)
                
                x_minus = np.zeros(n)
                x_minus[i] = -scale
                eval_point(x_minus)

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        x0 = best_x.copy()

        # --- Phase 2: CMA-ES ---
        # CMA-ES is robust to rotations and shifts.
        # We allocate a significant chunk of the remaining budget to it.
        remaining = budget - calls
        cma_budget = int(remaining * 0.7)
        
        # Ensure CMA-ES has enough budget to be effective.
        # CMA-ES typically needs ~ (10 + 2*sqrt(n)) * n evaluations for good convergence.
        # For n=5, that's ~35 * 5 = 175.
        # If remaining is small, we might skip or reduce.
        
        if cma_budget > 50:
            try:
                import cma
                # Use a wrapper that respects the budget and deadline
                def cma_func(z):
                    if time.time() > deadline or calls >= budget:
                        return 1e100 # Large penalty to stop CMA-ES effectively
                    return eval_point(z)
                
                # Initial sigma:
                # If we found a point close to the minimum, use a smaller sigma.
                # If not, use a larger one.
                # A safe bet is 1.0 or 2.0.
                sigma0 = 1.0
                
                # Bounds
                bounds = [(-5.0,) * n, (5.0,) * n]
                
                # CMA-ES options
                opts = {
                    'bounds': bounds,
                    'maxfevals': cma_budget,
                    'seed': 42,
                    'verbose': -1,
                    'quiet': True,
                    'pickles': False,
                    'tolfun': 1e-15,
                    'tolx': 1e-12,
                    'tolfun' : 1e-15,
                    'tolfun_interval': 1,
                    'tolfun_hist': 1e-15,
                    'tolx': 1e-12,
                    'tolx_interval': 1,
                    'tolfun': 1e-15,
                    'tolfun_interval': 1,
                    'tolfun_hist': 1e-15,
                    'tolfun': 1e-15,
                    'tolfun_interval': 1,
                    'tolfun_hist': 1e-15,
                    'tolfun': 1e-15,
                    'tolfun_interval': 1,
                    'tolfun_hist': 1e-15,
                }
                
                # Note: cma library uses 'bounds' as a tuple (lb, ub)
                res = cma.fmin(cma_func, x0, sigma0, opts)
                
                # cma.fmin returns (xbest, fbest, res)
                # Update best if CMA-ES found something better
                # Note: cma_func calls eval_point which updates best_f/best_x
                # So we don't strictly need to update here, but it's good practice.
                # However, cma might return a point that was not the absolute best in our history
                # if the penalty was triggered, but eval_point tracks the true best.
                
            except Exception:
                # If CMA-ES fails, fall back to L-BFGS-B or just return best so far
                pass

        if time.time() > deadline or calls >= budget:
            return best_x, best_f

        # --- Phase 3: L-BFGS-B Polish ---
        # Use L-BFGS-B to squeeze out the last bit of precision.
        # It is very efficient for smooth functions like Sphere.
        remaining = budget - calls
        if remaining > 10:
            try:
                from scipy.optimize import minimize
                
                def lbfgs_func(x):
                    if time.time() > deadline or calls >= budget:
                        return 1e100
                    return eval_point(x)
                
                bounds = [(-5.0, 5.0)] * n
                
                # Start from current best
                x_start = best_x.copy()
                
                res = minimize(lbfgs_func, x_start, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 1000,
                                        'maxfun': remaining,
                                        'gtol': 1e-15,
                                        'ftol': 1e-15})
                
                # eval_point already updated best_f and best_x
            except Exception:
                pass

        return best_x, best_f
