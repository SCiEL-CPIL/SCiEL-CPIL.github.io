"""Multi-start Differential Evolution with polish for BBOB Weierstrass f16."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    """Multi-start DE for the BBOB Weierstrass (f16) task."""

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
            except:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf
        
        # Phase 1: Differential Evolution for global exploration
        # DE uses a population; with popsize=15 in dim=5, that's 75 evaluations per generation
        # With ~500 budget, we can afford ~5-6 generations of DE
        # Use a smaller population to conserve budget for polish
        popsize = 12  # 12 * 5 = 60 evals per generation
        
        try:
            de_res = differential_evolution(
                safe_func,
                bounds=[(-5.0, 5.0)] * dim,
                maxiter=10,
                popsize=popsize,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                seed=42,
                polish=False,  # We'll do our own polish
                init='latinhypercube',
                update_interval=1,
                callback=lambda xk, convergence: (
                    stop_flag[0] or n_calls[0] >= budget - 50 or time.time() > deadline
                )
            )
            if de_res.fun < best_f and np.isfinite(de_res.fun):
                best_f = de_res.fun
                best_x = de_res.x.copy()
        except Exception:
            pass

        # Phase 2: Nelder-Mead polish from best DE result
        remaining = budget - n_calls[0]
        if remaining > 20 and not stop_flag[0] and time.time() < deadline:
            polish_calls = [0]
            def polish_func(x):
                if polish_calls[0] >= remaining or n_calls[0] >= budget or time.time() > deadline:
                    return np.inf
                polish_calls[0] += 1
                return safe_func(x)
            
            try:
                res = minimize(
                    polish_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 100,
                        'xatol': 1e-10, 
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: If we still have budget, try a few more NM restarts from random points
        remaining = budget - n_calls[0]
        if remaining > 30 and not stop_flag[0] and time.time() < deadline:
            rng = np.random.default_rng(123)
            n_extra_restarts = min(3, remaining // 30)
            for i in range(n_extra_restarts):
                if n_calls[0] >= budget or time.time() > deadline or stop_flag[0]:
                    break
                x0 = rng.uniform(-4.0, 4.0, size=dim)
                restart_calls = [0]
                calls_per_restart = min(30, budget - n_calls[0])
                def nm_func(x):
                    if restart_calls[0] >= calls_per_restart or n_calls[0] >= budget or time.time() > deadline:
                        return np.inf
                    restart_calls[0] += 1
                    return safe_func(x)
                
                try:
                    res = minimize(
                        nm_func, 
                        x0, 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': 50,
                            'xatol': 1e-8, 
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                    if res.fun < best_f and np.isfinite(res.fun):
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
