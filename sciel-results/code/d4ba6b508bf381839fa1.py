import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290.0  # Safety margin before 330s kill
        
        n = self.dim
        budget = self.budget
        
        eval_count = 0
        best_x = None
        best_f = np.inf
        
        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if time.time() - start_time > time_limit:
                return best_f if best_f < np.inf else 1e10
            if eval_count >= budget:
                return best_f if best_f < np.inf else 1e10
            
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                f_val = func(x_clipped)
            except Exception:
                return 1e10
                
            eval_count += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # Multi-start CMA-ES strategy
        # Use CMA-ES with multiple restarts to handle multimodality
        from scipy.optimize import minimize
        
        # Phase 1: Initial exploration with several random starting points
        # Use 10% of budget for initial exploration
        init_budget = max(10, budget // 10)
        init_points = []
        
        # Generate initial points using Latin hypercube-like sampling
        rng = np.random.RandomState(42)
        
        for i in range(init_budget):
            # Spread points across the domain
            x_init = rng.uniform(-5.0, 5.0, size=n)
            f_init = wrapped_func(x_init)
            init_points.append((x_init, f_init))
        
        # Sort by function value and keep top candidates
        init_points.sort(key=lambda x: x[1])
        top_starts = [x for x, f in init_points[:5]]
        
        if eval_count >= budget:
            return best_x, best_f
        
        # Phase 2: CMA-ES from multiple starting points
        # Use remaining budget for CMA-ES
        remaining_budget = budget - eval_count
        num_starts = min(3, len(top_starts))  # Use up to 3 best starting points
        
        for start_idx in range(num_starts):
            if eval_count >= budget:
                break
            
            if time.time() - start_time > time_limit:
                break
            
            x_start = top_starts[start_idx]
            
            # Set up CMA-ES parameters
            sigma0 = 2.0  # Initial step size
            
            # Use minimize with Nelder-Mead as a simpler alternative
            # CMA-ES would be better but Nelder-Mead is more reliable
            try:
                result = minimize(
                    wrapped_func,
                    x_start,
                    method='Nelder-Mead',
                    options={'maxiter': max(50, remaining_budget // (3 * n)), 
                             'xatol': 1e-6, 
                             'fatol': 1e-8,
                             'adaptive': True}
                )
            except Exception:
                pass
            
            if eval_count >= budget:
                break
        
        return best_x, best_f
