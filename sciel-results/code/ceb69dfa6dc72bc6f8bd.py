import numpy as np
import time
import scipy.optimize


class Optimizer:
    """
    Hybrid optimizer:
    1. Global phase: Multi-point (1+1)-ES with adaptive step size and restarts from best.
    2. Local phase: L-BFGS-B for efficient smooth convergence, followed by Nelder-Mead polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, use 280s to be safe
        deadline = start_time + 280
        
        n = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= self.budget:
                raise StopIteration("Budget exceeded")
            
            x_arr = np.asarray(x, dtype=np.float64)
            if x_arr.shape != (n,):
                x_arr = x_arr.reshape(-1)
                if x_arr.size != n:
                    raise ValueError("Wrong dimension")
            f_val = float(func(x_arr))
            calls += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_arr.copy()
            return f_val

        rng = np.random.default_rng(0)
        
        # Phase 1: Global Search with Multi-point (1+1)-ES and Restarts
        # Allocate 60% of budget to global exploration
        global_budget = int(self.budget * 0.60)
        
        # Multi-point initialization (5 points) for better early coverage
        num_starts = 5
        starts = []
        for i in range(num_starts):
            if i == 0:
                x = np.zeros(n)
            else:
                x = rng.uniform(-4, 4, n)
            try:
                fx = safe_func(x)
            except (TimeoutError, StopIteration):
                break
            starts.append((x.copy(), fx))
        
        if not starts:
            return best_x, best_f

        # Start from the best initial point
        best_start = min(starts, key=lambda t: t[1])
        x = best_start[0].copy()
        fx = best_start[1]
        
        sigma = 1.0
        stagnation = 0
        max_stagnation = 50
        
        # Track best in current run for restarts
        run_best_x = x.copy()
        run_best_f = fx

        for _ in range(global_budget):
            if calls >= self.budget or time.time() > deadline:
                break
                
            # Generate candidate
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            
            try:
                fy = safe_func(y)
            except (TimeoutError, StopIteration):
                break
                
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                if sigma > 4.0:
                    sigma = 4.0
                stagnation = 0
                if fx < run_best_f:
                    run_best_f = fx
                    run_best_x = x.copy()
            else:
                sigma *= 0.9
                if sigma < 0.05:
                    sigma = 0.05
                stagnation += 1
                
                # If stagnating, restart from global best with larger step
                if stagnation > max_stagnation:
                    if best_x is not None:
                        x = best_x.copy()
                        fx = best_f
                    else:
                        x = rng.uniform(-4, 4, n)
                        try:
                            fx = safe_func(x)
                        except (TimeoutError, StopIteration):
                            break
                    sigma = 2.0
                    stagnation = 0
                    # Reduce threshold for more aggressive restarts later
                    max_stagnation = max(15, max_stagnation // 2)

        # Phase 2: Local Refinement with L-BFGS-B
        # L-BFGS-B is more efficient for smooth functions and uses gradient approximations
        if calls < self.budget and time.time() < deadline and best_x is not None:
            remaining_calls = self.budget - calls
            # L-BFGS-B uses ~dim evaluations per iteration roughly, but with line search
            max_iter = max(10, remaining_calls // 2)
            
            try:
                result = scipy.optimize.minimize(
                    safe_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * n,
                    options={
                        'maxiter': max_iter,
                        'ftol': 1e-10,
                        'gtol': 1e-8,
                        'maxls': 20
                    }
                )
            except (TimeoutError, StopIteration, Exception):
                pass # Keep best from Phase 1 if this fails
        
        # Phase 3: Final Nelder-Mead polish if budget remains
        if calls < self.budget - 5 and time.time() < deadline and best_x is not None:
            remaining_calls = self.budget - calls
            if remaining_calls > 10:
                max_iter = max(5, remaining_calls // (n + 2))
                try:
                    result2 = scipy.optimize.minimize(
                        safe_func, 
                        best_x, 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': max_iter,
                            'xatol': 1e-6,
                            'fatol': 1e-6,
                            'adaptive': True
                        }
                    )
                except (TimeoutError, StopIteration, Exception):
                    pass

        return best_x, best_f
