import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        D = self.dim
        budget = self.budget
        
        call_count = [0]
        best_x = [np.zeros(D)]
        best_f = [np.inf]

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if call_count[0] >= budget:
                raise ValueError("Budget exceeded")
            
            call_count[0] += 1
            x = np.asarray(x, dtype=float).flatten()
            if x.shape[0] != D:
                x = x[:D] if len(x) > D else np.pad(x, (0, D - len(x)))
            
            f_val = float(func(x))
            if f_val < best_f[0]:
                best_f[0] = f_val
                best_x[0] = x.copy()
            return f_val

        # Phase 1: Global Search with CMA-ES
        # CMA-ES is robust to rotation and scaling.
        # Start from origin.
        try:
            # maxiter is high, but budget guard will stop it.
            # sigma0 is set to 1.0 to cover the space [-5, 5].
            res1 = minimize(
                wrapped_func, 
                np.zeros(D), 
                method='CMA-ES',
                options={
                    'maxiter': 1000,
                    'sigma0': 1.0,
                    'tol': 1e-6,
                    'seed': 0
                }
            )
            if res1.fun < best_f[0]:
                best_f[0] = res1.fun
                best_x[0] = res1.x
        except (TimeoutError, ValueError):
            pass
        except Exception:
            pass

        # Phase 2: Local Polish with CMA-ES
        # Use remaining budget for high-precision convergence.
        if call_count[0] < budget:
            try:
                start_x = best_x[0].copy()
                res2 = minimize(
                    wrapped_func, 
                    start_x, 
                    method='CMA-ES',
                    options={
                        'maxiter': 1000,
                        'sigma0': 1e-3, # Small step for polishing
                        'tol': 1e-12,
                        'seed': 1
                }
                )
                if res2.fun < best_f[0]:
                    best_f[0] = res2.fun
                    best_x[0] = res2.x
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        return (best_x[0], best_f[0])
