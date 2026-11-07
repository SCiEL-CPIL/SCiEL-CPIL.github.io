import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill
        
        rng = np.random.default_rng(42)
        
        # 1. Initial diverse sampling to find a good starting basin
        # Rosenbrock is narrow, so random uniform is often poor.
        # We do a small batch of random starts, pick the best, then polish.
        # Also include the origin and a few structured points.
        
        n_initial = 30
        X = np.vstack([
            np.zeros(self.dim),
            np.ones(self.dim) * 0.5,
            np.ones(self.dim) * -0.5,
            rng.uniform(-5, 5, (n_initial - 3, self.dim))
        ])
        F = np.array([func(x) for x in X])
        
        best_idx = np.argmin(F)
        best_x = X[best_idx].copy()
        best_f = F[best_idx]
        
        calls_used = n_initial
        remaining = self.budget - calls_used
        
        # Check time
        if time.time() > deadline - 5:
            return best_x, best_f
        
        # 2. Local Refinement Phase
        # Use L-BFGS-B for efficient convergence in the valley.
        # Rosenbrock is smooth, so gradient-based methods shine once in the basin.
        
        # Run L-BFGS-B with budget tracking and time checks.
        call_count = [0]
        
        def wrapper(x):
            if call_count[0] >= remaining:
                raise ValueError("Budget exceeded")
            if time.time() > deadline - 5:
                raise ValueError("Time exceeded")
            call_count[0] += 1
            return func(x)
        
        try:
            res = minimize(wrapper, best_x, method='L-BFGS-B', 
                           options={'maxiter': 500, 'maxfun': remaining, 'maxls': 15, 'gtol': 1e-6})
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                
        except Exception:
            # If L-BFGS-B fails or exceeds budget, we stick with the best found so far.
            pass
        
        remaining = self.budget - (calls_used + call_count[0])
        
        # Check time
        if time.time() > deadline - 10:
            return best_x, best_f
        
        # 3. If we have budget left, do additional local optimizations from perturbed starts
        # to explore nearby basins and potentially find a better minimum.
        
        if remaining > 30:
            n_restarts = min(5, remaining // 10)
            for i in range(n_restarts):
                if time.time() > deadline - 10:
                    break
                if remaining < 10:
                    break
                
                # Perturb the best point with adaptive scale
                scale = 0.3 + 0.2 * (i / max(1, n_restarts - 1))
                perturbation = rng.normal(0, scale, self.dim)
                start_x = np.clip(best_x + perturbation, -5, 5)
                
                restart_call_count = [0]
                
                def restart_wrapper(x):
                    if restart_call_count[0] >= remaining:
                        raise ValueError("Budget exceeded")
                    if time.time() > deadline - 5:
                        raise ValueError("Time exceeded")
                    restart_call_count[0] += 1
                    return func(x)
                
                try:
                    res_r = minimize(restart_wrapper, start_x, method='L-BFGS-B',
                                     options={'maxiter': 200, 'maxfun': remaining, 'maxls': 10})
                    
                    if res_r.fun < best_f:
                        best_f = res_r.fun
                        best_x = res_r.x.copy()
                        
                except Exception:
                    pass
                
                remaining = self.budget - (calls_used + call_count[0] + restart_call_count[0])
        
        # 4. Final high-precision polish with Nelder-Mead if budget allows
        if remaining > 15 and time.time() < deadline - 15:
            nm_call_count = [0]
            
            def nm_wrapper(x):
                if nm_call_count[0] >= remaining:
                    raise ValueError("Budget exceeded")
                if time.time() > deadline - 5:
                    raise ValueError("Time exceeded")
                nm_call_count[0] += 1
                return func(x)
            
            try:
                # Tighten the simplex around the current best
                step = 0.05
                x0 = best_x.copy()
                res_nm = minimize(nm_wrapper, x0, method='Nelder-Mead',
                                  options={'maxiter': 300, 'maxfev': remaining,
                                           'xatol': 1e-10, 'fatol': 1e-10,
                                           'initial_simplex': np.array([x0 + step * np.eye(self.dim)[i] for i in range(self.dim)] + [x0])})
                
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x.copy()
                    
            except Exception:
                pass
        
        return best_x, best_f
