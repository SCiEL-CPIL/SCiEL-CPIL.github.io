"""Optimizer for BBOB-style Rosenbrock (f8) on [-5,5]^5."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid optimizer: diverse multi-start scan + Nelder-Mead local refinement + L-BFGS-B polish."""

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

        # Phase 1: Diverse global scan with structured + random points
        # Rosenbrock valley structure: minimum near (1,...,1) in standard form,
        # but hidden shift/rotation means we need broad coverage.
        # Use structured points at ±1, ±2, ±3, ±4 plus random samples.
        
        n_scan = min(40, budget // 4)  # up to 40 scan points
        scan_points = []
        
        # Structured points: combinations of {-4,-3,-2,-1,0,1,2,3,4}
        # Include origin and key structured points
        structured = [np.zeros(dim)]
        
        # Add points with 1-2 nonzero coordinates at structured values
        vals = [1, 2, 3, 4, -1, -2, -3, -4]
        for v in vals:
            for i in range(dim):
                x = np.zeros(dim)
                x[i] = v
                structured.append(x)
        
        # Add some 2-coordinate points
        for v1 in [2, 3, -2, -3]:
            for v2 in [2, 3, -2, -3]:
                for i in range(dim):
                    for j in range(i+1, dim):
                        x = np.zeros(dim)
                        x[i] = v1
                        x[j] = v2
                        structured.append(x)
        
        # Shuffle and take up to n_scan structured points
        rng.shuffle(structured)
        for sp in structured[:n_scan]:
            if calls >= budget or time.time() > deadline:
                break
            f_try = safe_call(sp)
            scan_points.append((sp, f_try))
        
        # Fill remaining scan budget with random points
        n_random = n_scan - len(scan_points)
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x_try = rng.uniform(-4.5, 4.5, size=dim)
            f_try = safe_call(x_try)
            scan_points.append((x_try, f_try))
        
        # Find top K promising starts
        if scan_points:
            scan_points.sort(key=lambda t: t[1])
            top_k = min(5, len(scan_points))
            promising_starts = [scan_points[i][0] for i in range(top_k)]
        else:
            promising_starts = [np.zeros(dim)]

        # Phase 2: Nelder-Mead local refinement from each promising start
        # Nelder-Mead navigates Rosenbrock valleys well by maintaining directional info.
        remaining_budget = budget - calls
        if remaining_budget > 30:
            # Allocate budget across top starts, giving more to better starts
            nm_budgets = []
            total_nm = remaining_budget - 50  # reserve 50 for final polish
            if total_nm > 20:
                # Give 50% to best, 25% to second, etc.
                weights = [0.5, 0.25, 0.15, 0.075, 0.025]
                for i in range(min(len(promising_starts), 5)):
                    nm_budgets.append(int(total_nm * weights[i]))
                # Ensure at least 20 for the best
                if nm_budgets and nm_budgets[0] < 20:
                    nm_budgets[0] = min(20, total_nm)
            
            for i, x_start in enumerate(promising_starts):
                if calls >= budget or time.time() > deadline:
                    break
                if i >= len(nm_budgets) or nm_budgets[i] < 10:
                    continue
                
                maxfev_nm = nm_budgets[i]
                
                try:
                    res_nm = minimize(
                        safe_call, 
                        x_start.copy(), 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': 200, 
                            'maxfev': maxfev_nm, 
                            'xatol': 1e-5, 
                            'fatol': 1e-10,
                            'adaptive': True,
                            'disp': False
                        }
                    )
                    if res_nm.fun < best_f:
                        best_x = res_nm.x.copy()
                        best_f = res_nm.fun
                except Exception:
                    pass

        # Phase 3: Final high-precision L-BFGS-B polish
        remaining_budget = budget - calls
        if remaining_budget > 30:
            # L-BFGS-B with central differences: ~2*dim + 1 = 11 calls per iteration
            # Allow tight tolerances for high precision
            maxiter_lbfgs = max(10, min(100, (remaining_budget - 5) // 12))
            
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
                        'ftol': 1e-15, 
                        'gtol': 1e-12, 
                        'disp': False
                    }
                )
                if res_lbfgs.fun < best_f:
                    best_x = res_lbfgs.x.copy()
                    best_f = res_lbfgs.fun
            except Exception:
                pass
        
        return best_x, best_f
