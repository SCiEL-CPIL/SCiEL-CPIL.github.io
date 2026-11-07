import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin before 300s limit

        def time_left():
            return deadline - time.time()

        calls = 0
        best_x = None
        best_f = np.inf
        all_points = []
        all_vals = []

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls += 1
            all_points.append(x.copy())
            all_vals.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global exploration using Sobol-like quasi-random sequence
        # This provides better coverage of the 5D space than pure random sampling
        # for early error reduction (AOCC benefits from rapid initial progress)
        
        # First, evaluate a few deterministic points for immediate progress
        deterministic_points = [
            np.zeros(dim),
            np.ones(dim) * 0.5,
            np.ones(dim) * -0.5,
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
        ]
        
        for p in deterministic_points:
            if calls >= budget or time_left() < 10.0:
                break
            evaluate(p)
        
        # Generate Sobol-like quasi-random points for broad exploration
        # Use a simple low-discrepancy sequence approach
        n_global = min(150, budget - calls - 100)  # Leave budget for local refinement
        
        # Simple quasi-random generation using bit-reversal (Sobol-like)
        def sobol_like(index, dim):
            point = np.zeros(dim)
            for d in range(dim):
                # Van der Corput sequence for each dimension
                val = 0.0
                denom = 1.0
                n = index + 1
                while n > 0:
                    denom *= (d + 2)  # Different base for each dimension
                    val += (n % (d + 2)) / denom
                    n //= (d + 2)
                point[d] = val
            return point

        for i in range(n_global):
            if calls >= budget or time_left() < 10.0:
                break
            x = sobol_like(i, dim)
            x = -5.0 + 10.0 * x  # Map to [-5, 5]^5
            evaluate(x)
        
        # Also add some random points for diversity
        n_random = min(30, budget - calls - 50)
        for _ in range(n_random):
            if calls >= budget or time_left() < 10.0:
                break
            x = np.random.uniform(-5, 5, dim)
            evaluate(x)
        
        # Phase 2: Local refinement using (1+1)-ES from the best point
        # This combines the global exploration with adaptive local search
        
        if best_x is not None and calls < budget - 20 and time_left() > 10.0:
            x = best_x.copy()
            f = best_f
            sigma = 0.5  # Start with smaller step size for local refinement
            
            # 1/5 success rule parameters
            success_count = 0
            step_count = 0
            sigma_factor_up = 1.1
            sigma_factor_down = 1.0 / 1.1
            target_success_rate = 0.2
            
            while calls < budget and time_left() > 1.0:
                # Generate candidate
                z = np.random.standard_normal(dim)
                x_new = x + sigma * z
                
                # Evaluate
                f_new = evaluate(x_new)
                
                # Update best
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                
                # 1/5 success rule
                if f_new < f:
                    success_count += 1
                    x = x_new
                    f = f_new
                else:
                    pass
                
                step_count += 1
                
                # Adapt sigma every 10 steps
                if step_count >= 10:
                    success_rate = success_count / step_count
                    if success_rate > target_success_rate:
                        sigma *= sigma_factor_up
                    else:
                        sigma *= sigma_factor_down
                    
                    # Reset counters
                    success_count = 0
                    step_count = 0
                    
                    # Clamp sigma to reasonable range
                    sigma = max(0.001, min(sigma, 5.0))
                    
                    # If sigma is very small, we've likely converged
                    if sigma < 0.01:
                        break
        
        # Phase 3: If budget remains, try Nelder-Mead from the best point
        if best_x is not None and calls < budget - 10 and time_left() > 5.0:
            try:
                result = minimize(
                    func, best_x, method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-10,
                        'fatol': 1e-12
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
