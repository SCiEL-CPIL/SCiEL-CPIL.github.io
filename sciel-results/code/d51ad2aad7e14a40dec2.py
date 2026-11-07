"""Multi-start local search using SciPy Nelder-Mead for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start local optimizer for Schaffers F7 on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320
        dim = self.dim
        budget = self.budget

        calls = [0]
        
        def eval_func(x):
            if calls[0] >= budget or time.time() > deadline:
                raise RuntimeError("Budget or time exceeded")
            calls[0] += 1
            return func(np.clip(x, -5, 5))

        def make_simplex(x0, step=1.0):
            simplex = [x0.copy()]
            for i in range(dim):
                pt = x0.copy()
                pt[i] += step
                simplex.append(pt)
            return np.array(simplex)

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # Phase 1: Multi-start local search
        # 10 restarts with limited iterations
        n_restarts = 10
        max_iter_restart = 10

        for i in range(n_restarts):
            if calls[0] >= budget or time.time() > deadline:
                break
            
            x0 = rng.uniform(-4.5, 4.5, dim)
            try:
                simplex = make_simplex(x0)
                res = minimize(eval_func, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter_restart, 
                                        'initial_simplex': simplex,
                                        'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except RuntimeError:
                break
            except Exception:
                break

        # Phase 2: Final polish
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            try:
                remaining = budget - calls[0]
                # Estimate max iterations for polish
                # Nelder-Mead cost ~ (dim+1) per iter
                max_iter_polish = max(10, remaining // (dim + 1))
                
                simplex = make_simplex(best_x, step=0.1)
                res = minimize(eval_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': max_iter_polish,
                                        'initial_simplex': simplex,
                                        'xatol': 1e-12, 'fatol': 1e-12})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except RuntimeError:
                pass
            except Exception:
                pass

        if best_x is None:
            # Fallback
            try:
                x0 = np.zeros(dim)
                best_f = eval_func(x0)
                best_x = x0
            except:
                best_x = np.zeros(dim)
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
