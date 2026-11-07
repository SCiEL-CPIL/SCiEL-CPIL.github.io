import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        def counting_func(x):
            nonlocal eval_count
            if eval_count >= budget:
                return np.inf
            eval_count += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)
        
        # Generate diverse start points using Latin Hypercube Sampling
        # 4 starts for dim=5 with 500 budget: ~125 evals per start
        num_starts = 4
        
        def lhs_sample(n, dim, rng):
            """Latin Hypercube Sampling"""
            samples = np.zeros((n, dim))
            for d in range(dim):
                perm = rng.permutation(n)
                samples[:, d] = (perm + rng.random(n)) / n
            return samples
        
        rng = np.random.default_rng(42)
        lhs_points = lhs_sample(num_starts, dim, rng)
        # Map [0,1] to [-5,5]
        start_points = lhs_points * 10.0 - 5.0
        
        # Add origin as an extra candidate
        start_points = np.vstack([start_points, np.zeros(dim)])
        
        # CMA-ES parameters
        popsize = 8  # Smaller popsize for faster convergence per eval
        base_sigma = 1.5  # Larger initial step for better exploration
        
        # Adaptive budget allocation: start with ~100 evals, increase if improving
        min_budget_per_start = 80
        max_budget_per_start = 200
        
        # Track progress to decide if we should continue
        prev_best_f = np.inf
        
        for i in range(len(start_points)):
            if eval_count >= budget or time.time() > deadline:
                break
                
            x0 = start_points[i]
            
            # Determine budget for this start
            remaining = budget - eval_count
            if i < len(start_points) - 1:
                # Not the last start: use moderate budget
                alloc = min(min_budget_per_start + 20 * i, remaining)
                alloc = max(alloc, 30)  # Minimum 30 evals
            else:
                # Last start: use all remaining
                alloc = remaining
            
            # Scale sigma based on start index (earlier starts get larger sigma)
            sigma = base_sigma * (1.0 - 0.1 * i)
            
            max_iter = max(1, alloc // popsize)
            
            try:
                res = scipy.optimize.cma(
                    counting_func,
                    x0,
                    sigma0=sigma,
                    options={
                        'maxiter': max_iter,
                        'popsize': popsize,
                        'tol': 1e-7,
                        'verbose': 0
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                # Fallback: simple local search
                try:
                    res_nm = scipy.optimize.minimize(
                        counting_func,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6}
                    )
                    if res_nm.fun < best_f:
                        best_f = res_nm.fun
                        best_x = res_nm.x.copy()
                except Exception:
                    pass
        
        # Final polish: CMA-ES from best point with small sigma
        if eval_count < budget and best_x is not None and time.time() < deadline:
            remaining = budget - eval_count
            if remaining > 20:
                try:
                    polish_sigma = 0.3
                    max_iter_polish = max(1, remaining // popsize)
                    
                    res_polish = scipy.optimize.cma(
                        counting_func,
                        best_x,
                        sigma0=polish_sigma,
                        options={
                            'maxiter': max_iter_polish,
                            'popsize': max(4, popsize // 2),
                            'tol': 1e-9,
                            'verbose': 0
                        }
                    )
                    
                    if res_polish.fun < best_f:
                        best_f = res_polish.fun
                        best_x = res_polish.x.copy()
                except Exception:
                    pass
        
        # Ensure we return something valid
        if best_x is None:
            best_x = np.zeros(dim)
            if eval_count < budget:
                best_f = counting_func(best_x)
            else:
                best_f = func(best_x)
                eval_count += 1

        return best_x, best_f
