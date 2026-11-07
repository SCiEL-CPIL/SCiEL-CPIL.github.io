import numpy as np
import time

class Optimizer:
    """
    Combined strategy for BBOB f21 (Gallagher 101 peaks).
    
    Strategy:
    1. Global exploration with diverse starting points (Latin Hypercube Sampling).
    2. For the most promising starting points, run Nelder-Mead local optimization.
    3. Use remaining budget for refinement with adaptive (1+1)-ES around the best point.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s. We stop at 310s to be safe.
        deadline = start_time + 310.0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            try:
                val = float(func(x))
            except Exception:
                return None
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(12345)

        # Phase 1: Global exploration - diverse starting points
        # Use 30% of budget for global search
        n_starts = max(15, min(60, budget // 3))
        
        # Latin Hypercube Sampling for diverse starting points
        starts = []
        for i in range(n_starts):
            if not check_time() or calls_made >= budget:
                break
            x = np.empty(dim)
            for d in range(dim):
                perm = rng.permutation(n_starts)
                x[d] = -5.0 + 10.0 * (perm[i] + rng.random()) / n_starts
            x = np.clip(x, -5, 5)
            fx = safe_call(x)
            if fx is None:
                break
            starts.append((fx, x.copy()))

        if best_x is None:
            # Fallback
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is not None:
                best_f = fx
                best_x = x.copy()
                starts = [(fx, x.copy())]

        # Sort starting points by function value (best first)
        starts.sort(key=lambda t: t[0])

        # Phase 2: Local optimization using Nelder-Mead for each promising start
        # Use remaining budget wisely
        budget_per_local = max(10, (budget - calls_made) // max(1, len(starts[:10])))
        
        for i, (f0, x0) in enumerate(starts[:10]):
            if not check_time() or calls_made >= budget:
                break
            
            # Skip if this start is already worse than best by a lot
            if f0 > best_f + 50:
                continue
                
            # Run Nelder-Mead local optimization
            try:
                from scipy.optimize import minimize
                result = minimize(
                    func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': budget_per_local,
                        'maxfev': budget_per_local,
                        'adaptive': True
                    }
                )
                if result.x is not None:
                    fx = safe_call(result.x)
                    if fx is not None and fx < best_f:
                        best_f = fx
                        best_x = result.x.copy()
            except Exception:
                pass

        # Phase 3: Refinement around the best point found
        if best_x is not None and check_time() and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            
            # Adaptive (1+1)-ES for final refinement
            sigma = 0.5
            no_improve = 0
            max_no_improve = 100
            total_steps = 0
            success_count = 0
            window = 10
            
            while check_time() and calls_made < budget:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                
                total_steps += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    no_improve = 0
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                    no_improve += 1
                
                if total_steps >= window:
                    success_rate = success_count / total_steps
                    if success_rate > 0.25:
                        sigma *= 1.15
                    elif success_rate < 0.15:
                        sigma *= 0.85
                    success_count = 0
                    total_steps = 0
                
                sigma = max(sigma, 1e-5)
                sigma = min(sigma, 3.0)
                
                if no_improve > max_no_improve:
                    break

        return best_x, best_f
