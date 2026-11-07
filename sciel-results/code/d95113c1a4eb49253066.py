import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0
        eval_history = []  # (fx, x) for all evaluations

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            eval_history.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def run_nm(x0, max_iter, xatol=1e-8, fatol=1e-10):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def objective(x):
                    return evaluate(x)
                
                minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': xatol,
                        'fatol': fatol,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        def run_powell(x0, maxiter, ftol=1e-8):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def objective(x):
                    return evaluate(x)
                
                minimize(
                    objective,
                    x0,
                    method='Powell',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': maxiter,
                        'ftol': ftol
                    }
                )
            except Exception:
                pass

        # Phase 1: Dense Sobol quasi-random exploration
        # Use 350 evaluations for systematic coverage of the search space
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=n, scramble=True, seed=42)
        num_sobol = 350
        sobol_points = sobol.random_base2(m=9)  # 512 points, take first 350
        sobol_points = sobol_points[:num_sobol]
        # Map [0,1]^n to [-5,5]^n
        sobol_points = sobol_points * 10.0 - 5.0
        
        for x in sobol_points:
            if calls >= budget or time.time() >= deadline:
                break
            evaluate(x)
        
        # Phase 2: Identify top candidates from exploration
        # Sort by function value and pick diverse top points
        if eval_history and calls < budget and time.time() < deadline:
            # Sort by function value
            sorted_hist = sorted(eval_history, key=lambda t: t[0])
            top_k = min(5, len(sorted_hist))
            
            # Select diverse top candidates (spread out in space)
            selected = []
            for fx, x in sorted_hist:
                # Check if this point is sufficiently different from already selected
                is_diverse = True
                for _, sx in selected:
                    if np.linalg.norm(x - sx) < 1.0:
                        is_diverse = False
                        break
                if is_diverse:
                    selected.append((fx, x))
                if len(selected) >= top_k:
                    break
            
            # If not enough diverse points, fill with remaining top
            if len(selected) < top_k:
                for fx, x in sorted_hist:
                    if (fx, x) not in selected:
                        selected.append((fx, x))
                    if len(selected) >= top_k:
                        break
            
            # Phase 3: Local optimization from each top candidate
            for fx, x in selected:
                if calls >= budget or time.time() >= deadline:
                    break
                run_powell(x, maxiter=20, ftol=1e-8)
                if calls < budget and time.time() < deadline:
                    run_nm(x, max_iter=15, xatol=1e-9, fatol=1e-11)
        
        # Phase 4: Final high-precision polish on the best point
        if best_x is not None and calls < budget and time.time() < deadline:
            run_nm(best_x, max_iter=30, xatol=1e-11, fatol=1e-13)

        return best_x, best_f
