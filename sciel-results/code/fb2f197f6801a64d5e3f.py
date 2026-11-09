import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        
        best_x = None
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)
        
        def wrapped_func(x):
            nonlocal best_x, best_f, call_count
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if call_count >= self.budget:
                raise StopIteration("Budget exceeded")
            
            x = np.asarray(x, dtype=np.float64).flatten()
            if x.shape != (self.dim,):
                x = x[:self.dim]
                if x.shape[0] < self.dim:
                    x = np.pad(x, (0, self.dim - x.shape[0]))
            
            fx = func(x)
            call_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            return fx

        bounds = [(-5.0, 5.0)] * self.dim
        
        # Phase 1: Initial random samples to get a starting point
        n_init = min(20, self.budget // 10)
        for i in range(n_init):
            if time.time() > deadline or call_count >= self.budget:
                break
            x = rng.uniform(-4.0, 4.0, size=self.dim)
            try:
                wrapped_func(x)
            except (TimeoutError, StopIteration):
                break
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            try:
                wrapped_func(best_x)
            except (TimeoutError, StopIteration):
                pass
        
        # Phase 2: Differential evolution for global search
        remaining = self.budget - call_count
        if remaining > 50 and time.time() < deadline:
            maxiter_de = max(10, min(100, remaining // 20))
            popsize = max(10, min(20, remaining // 20))
            
            try:
                de_result = differential_evolution(
                    wrapped_func,
                    bounds=bounds,
                    maxiter=maxiter_de,
                    popsize=popsize,
                    tol=1e-6,
                    seed=42,
                    init='latinhypercube',
                    polish=False,
                    update_interval=1,
                    callback=lambda xk, convergence: convergence or time.time() > deadline
                )
            except Exception:
                pass
            
            if hasattr(de_result, 'x') and de_result.x is not None:
                try:
                    fx_de = wrapped_func(de_result.x)
                    if fx_de < best_f:
                        best_f = fx_de
                        best_x = de_result.x.copy()
                except (TimeoutError, StopIteration):
                    pass
        
        # Phase 3: Local refinement with L-BFGS-B from multiple starting points
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline:
            # Try L-BFGS-B from the current best and a few nearby points
            start_points = [best_x.copy()]
            
            # Add a few perturbed points around the best
            for _ in range(3):
                if call_count >= self.budget or time.time() > deadline:
                    break
                perturbation = 0.5 * rng.standard_normal(self.dim)
                xp = np.clip(best_x + perturbation, -5.0, 5.0)
                try:
                    fx_p = wrapped_func(xp)
                    start_points.append(xp)
                except (TimeoutError, StopIteration):
                    break
            
            for sp in start_points:
                if call_count >= self.budget or time.time() > deadline:
                    break
                
                try:
                    local_result = minimize(
                        wrapped_func,
                        sp,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'maxfun': max(10, (self.budget - call_count) // 3)}
                    )
                    
                    if local_result.x is not None and local_result.fun < best_f:
                        best_f = local_result.fun
                        best_x = local_result.x.copy()
                except (TimeoutError, StopIteration):
                    break
                except Exception:
                    continue
        
        return best_x, best_f
