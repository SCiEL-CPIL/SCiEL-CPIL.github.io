import time
import numpy as np
from scipy.optimize import differential_evolution

class Optimizer:
    """
    Hybrid DE + (1+1)-ES for BBOB Weierstrass (f16).
    Phase 1: Differential Evolution for global basin identification.
    Phase 2: (1+1)-ES with adaptive step size for local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        n_calls = [0]
        stop_flag = [False]

        def safe_func(x):
            if stop_flag[0] or n_calls[0] >= budget or time.time() > deadline:
                stop_flag[0] = True
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            try:
                return float(func(x))
            except Exception:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Differential Evolution
        # Use a fraction of the budget for global search
        de_budget = min(100, budget // 5)
        
        if de_budget > 10 and time.time() < deadline:
            bounds = [(-5.0, 5.0) for _ in range(dim)]
            start_calls = n_calls[0]
            
            # Use a small population to keep overhead low and calls high
            # popsize=10 means 10 calls per generation (plus initial)
            # We want to use ~de_budget calls.
            
            try:
                # DE callback to stop when budget is hit
                def de_callback(xk, convergence):
                    if stop_flag[0] or n_calls[0] - start_calls >= de_budget or time.time() > deadline:
                        stop_flag[0] = True
                        return True
                    return False

                res = differential_evolution(
                    safe_func,
                    bounds,
                    maxiter=100,
                    popsize=10,
                    tol=1e-12,
                    seed=42,
                    init='latinhypercube',
                    callback=de_callback
                )
                
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                # If DE fails, fall back to random
                pass

        # If DE didn't find anything or failed, do a quick random search
        if not np.isfinite(best_f) and n_calls[0] < budget:
             rng = np.random.default_rng(42)
             for _ in range(10):
                 if n_calls[0] >= budget or time.time() > deadline:
                     break
                 x = rng.uniform(-5.0, 5.0, size=dim)
                 f_x = safe_func(x)
                 if f_x < best_f:
                     best_f = f_x
                     best_x = x.copy()

        # Phase 2: (1+1)-ES Polish
        # Use remaining budget
        remaining = budget - n_calls[0]
        if remaining > 5 and time.time() < deadline and np.isfinite(best_f):
            x = best_x.copy()
            f_x = best_f
            
            # Initialize sigma based on problem scale
            # Weierstrass has features at small scales, but global min is in [-4,4]
            # Start with a moderate sigma
            sigma = 0.5
            
            rng = np.random.default_rng(123)
            c_sigma = 1.0 / (5 * dim)
            n_success = 0
            n_total = 0
            window_size = 5 * dim + 1
            
            while n_calls[0] < budget and time.time() < deadline and not stop_flag[0]:
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5.0, 5.0)
                
                f_trial = safe_func(x_trial)
                
                if not np.isfinite(f_trial):
                    break

                n_total += 1
                if f_trial < f_x:
                    n_success += 1
                    x = x_trial
                    f_x = f_trial
                    if f_x < best_f:
                        best_f = f_x
                        best_x = x_trial.copy()
                else:
                    # If no improvement, we might want to shrink sigma faster?
                    # The 1/5 rule handles this, but let's ensure it's responsive.
                    pass

                if n_total >= window_size:
                    current_sr = n_success / n_total
                    # Standard 1/5 rule
                    sigma *= np.exp(c_sigma * (current_sr - 0.5))
                    sigma = np.clip(sigma, 1e-6, 5.0)
                    n_success = 0
                    n_total = 0

        return best_x, best_f
