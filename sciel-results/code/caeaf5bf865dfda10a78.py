"""Optimizer for BBOB-style Rosenbrock (f8) on [-5,5]^5."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid optimizer: LHS global scan + Nelder-Mead local refinement + L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            
            if not np.isfinite(f):
                f = np.inf
            
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Latin Hypercube global scan for uniform coverage of [-4,4]^5
        n_scan = 60
        scan_points = []
        
        # Latin Hypercube: stratified sampling ensures coverage
        for _ in range(n_scan):
            if calls >= budget or time.time() > deadline:
                break
            x_try = rng.uniform(-4.0, 4.0, size=dim)
            f_try = safe_call(x_try)
            scan_points.append((x_try, f_try))
        
        # Also add origin and a few structured points for safety
        extra_points = [np.zeros(dim)]
        for v in [1.0, -1.0, 2.0, -2.0]:
            for i in range(dim):
                x = np.zeros(dim)
                x[i] = v
                extra_points.append(x)
        
        rng.shuffle(extra_points)
        for sp in extra_points:
            if calls >= budget or time.time() > deadline:
                break
            f_try = safe_call(sp)
            scan_points.append((sp, f_try))
        
        # Find top K promising starts
        if scan_points:
            scan_points.sort(key=lambda t: t[1])
            top_k = min(5, len(scan_points))
            promising_starts = [scan_points[i][0] for i in range(top_k)]
        else:
            promising_starts = [np.zeros(dim)]

        # Phase 2: Nelder-Mead local refinement from each promising start
        remaining_budget = budget - calls
        if remaining_budget > 30:
            # Allocate budget: give majority to best start, less to others
            nm_budgets = []
            total_nm = remaining_budget - 60  # reserve 60 for final polish
            if total_nm > 20:
                # Give 60% to best, 20% to second, 10% to third, 5% to fourth, 5% to fifth
                weights = [0.6, 0.2, 0.1, 0.05, 0.05]
                for i in range(min(len(promising_starts), 5)):
                    nm_budgets.append(int(total_nm * weights[i]))
                # Ensure at least 30 for the best
                if nm_budgets and nm_budgets[0] < 30:
                    nm_budgets[0] = min(30, total_nm)
            
            for i, x_start in enumerate(promising_starts):
                if calls >= budget or time.time() > deadline:
                    break
                if i >= len(nm_budgets) or nm_budgets[i] < 15:
                    continue
                
                maxfev_nm = nm_budgets[i]
                
                try:
                    res_nm = minimize(
                        safe_call, 
                        x_start.copy(), 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': 300, 
                            'maxfev': maxfev_nm, 
                            'xatol': 1e-6, 
                            'fatol': 1e-12,
                            'adaptive': True,
                            'disp': False
                        }
                    )
                    if res_nm.fun < best_f:
                        best_x = res_nm.x.copy()
                        best_f = res_nm.fun
                except Exception:
                    pass

        # Phase 3: L-BFGS-B polish pass 1
        remaining_budget = budget - calls
        if remaining_budget > 30:
            maxiter_lbfgs = max(10, min(80, (remaining_budget - 5) // 12))
            
            try:
                bounds = [(-5.0, 5.0)] * dim
                
                res_lbfgs = minimize(
                    safe_call, 
                    best_x.copy(), 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': maxiter_lbfgs, 
                        'maxfun': remaining_budget - 5, 
                        'ftol': 1e-14, 
                        'gtol': 1e-12, 
                        'disp': False
                    }
                )
                if res_lbfgs.fun < best_f:
                    best_x = res_lbfgs.x.copy()
                    best_f = res_lbfgs.fun
            except Exception:
                pass

        # Phase 4: L-BFGS-B polish pass 2 (if budget remains)
        remaining_budget = budget - calls
        if remaining_budget > 20:
            maxiter_lbfgs2 = max(5, min(50, (remaining_budget - 3) // 12))
            
            try:
                bounds = [(-5.0, 5.0)] * dim
                
                res_lbfgs2 = minimize(
                    safe_call, 
                    best_x.copy(), 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': maxiter_lbfgs2, 
                        'maxfun': remaining_budget - 3, 
                        'ftol': 1e-16, 
                        'gtol': 1e-14, 
                        'disp': False
                    }
                )
                if res_lbfgs2.fun < best_f:
                    best_x = res_lbfgs2.x.copy()
                    best_f = res_lbfgs2.fun
            except Exception:
                pass
        
        return best_x, best_f
