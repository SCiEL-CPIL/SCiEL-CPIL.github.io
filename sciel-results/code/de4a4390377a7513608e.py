"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """Multi-start optimizer: Differential Evolution + L-BFGS-B polish for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        best_x = np.zeros(n)
        best_fx = np.inf
        call_count = 0

        def safe_eval(x):
            nonlocal call_count, best_x, best_fx
            if call_count >= budget:
                return np.inf
            x = np.asarray(x, dtype=float).flatten()
            if x.shape != (n,) or not np.all(np.isfinite(x)):
                return np.inf
            x = np.clip(x, -5, 5)
            val = func(x)
            call_count += 1
            if not np.isfinite(val):
                return np.inf
            if val < best_fx:
                best_fx = val
                best_x = x.copy()
            return val

        def time_check():
            return time.time() > deadline

        # Phase 1: Differential Evolution for global exploration
        # Use a moderate population and maxiter to stay within budget
        de_budget = min(300, budget // 2)
        
        def de_func(x):
            if call_count >= budget:
                return 1e10
            return safe_eval(x)

        try:
            if budget > 50 and not time_check():
                # Differential evolution with tight bounds
                # For 5D, popsize=10 gives 50 per generation, maxiter=5 -> 250 calls
                popsize = max(5, min(15, de_budget // (n + 1) - 1))
                maxiter = max(2, de_budget // (popsize * n) - 1)
                
                res_de = differential_evolution(
                    de_func,
                    bounds=[(-5, 5)] * n,
                    popsize=popsize,
                    maxiter=maxiter,
                    seed=42,
                    tol=1e-10,
                    atol=1e-12,
                    polish=False,  # We'll do our own polish
                    init='latinhypercube',
                    maxfeval=de_budget
                )
                if np.isfinite(res_de.fun) and res_de.fun < best_fx:
                    best_fx = res_de.fun
                    best_x = np.clip(res_de.x, -5, 5).copy()
        except Exception:
            pass

        # Phase 2: Local polish with L-BFGS-B using finite differences
        # L-BFGS-B is efficient for smooth functions like Rosenbrock
        if not time_check() and call_count < budget:
            # Try polishing from the best point found
            polish_starts = [best_x.copy()]
            
            # Add a few random perturbations around the best point for robustness
            for _ in range(3):
                perturb = best_x + rng.normal(0, 0.1, n)
                perturb = np.clip(perturb, -5, 5)
                polish_starts.append(perturb)
            
            for start_x in polish_starts:
                if time_check() or call_count >= budget:
                    break
                
                def lbfgs_func(x):
                    if call_count >= budget:
                        return 1e10
                    return safe_eval(x)
                
                try:
                    res_lbfgs = minimize(
                        lbfgs_func,
                        start_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * n,
                        options={
                            'maxiter': 200,
                            'ftol': 1e-12,
                            'gtol': 1e-8,
                            'maxfun': budget - call_count
                        }
                    )
                    if np.isfinite(res_lbfgs.fun) and res_lbfgs.fun < best_fx:
                        best_fx = res_lbfgs.fun
                        best_x = np.clip(res_lbfgs.x, -5, 5).copy()
                except Exception:
                    pass

        # Phase 3: Final Nelder-Mead polish if budget remains
        if not time_check() and call_count < budget - 20:
            try:
                def nm_func(x):
                    if call_count >= budget:
                        return 1e10
                    return safe_eval(x)
                
                res_nm = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'maxfev': budget - call_count
                    }
                )
                if np.isfinite(res_nm.fun) and res_nm.fun < best_fx:
                    best_fx = res_nm.fun
                    best_x = np.clip(res_nm.x, -5, 5).copy()
            except Exception:
                pass

        # Ensure valid return
        if best_x is None or not np.all(np.isfinite(best_x)):
            best_x = np.zeros(n)
            best_fx = safe_eval(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
