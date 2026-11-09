import scipy
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= budget:
                raise StopIteration("Budget exceeded")
            
            x = np.asarray(x)
            f = func(x)
            calls += 1
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Global Search with (1+1)-ES and Restarts
        # Use 75% of budget for global exploration - more aggressive for multi-modal landscape
        global_budget = int(budget * 0.75)
        
        # Initial point
        x = rng.uniform(-5, 5, dim)
        try:
            fx = safe_func(x)
        except (TimeoutError, StopIteration):
            return best_x, best_f
            
        sigma = 1.0
        stagnation = 0
        max_stagnation = 25
        min_sigma = 0.005
        max_sigma = 10.0
        # Success-rate based sigma adaptation
        success_count = 0
        total_steps = 0
        window_size = 10

        for _ in range(global_budget):
            if calls >= budget or time.time() > deadline:
                break
                
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            
            try:
                fy = safe_func(y)
            except (TimeoutError, StopIteration):
                break
                
            total_steps += 1
            if fy < fx:
                x, fx = y, fy
                success_count += 1
                stagnation = 0
            else:
                stagnation += 1
                
                # If stagnating, restart
                if stagnation > max_stagnation:
                    # Restart from best point with higher probability
                    if best_x is not None and rng.random() < 0.7:
                        x = best_x.copy()
                        fx = best_f
                    else:
                        x = rng.uniform(-5, 5, dim)
                        try:
                            fx = safe_func(x)
                        except (TimeoutError, StopIteration):
                            break
                    
                    # Reset parameters for fresh start
                    sigma = 2.0
                    stagnation = 0
                    success_count = 0
                    total_steps = 0
                    max_stagnation = max(10, max_stagnation // 2)
                    
                    continue  # Skip sigma adaptation this iteration
            
            # Success-rate based sigma adaptation
            if total_steps >= window_size:
                success_rate = success_count / total_steps
                # Target ~35% success rate for optimal exploration
                if success_rate > 0.45:
                    sigma = min(sigma * 1.15, max_sigma)
                elif success_rate < 0.25:
                    sigma = max(sigma * 0.85, min_sigma)
                else:
                    sigma = min(sigma * 1.02, max_sigma)
                success_count = 0
                total_steps = 0

        # Phase 2: Local Refinement using Nelder-Mead
        # Nelder-Mead is derivative-free and robust for non-smooth functions like Schwefel
        if calls < budget and time.time() < deadline and best_x is not None:
            remaining_calls = budget - calls
            if remaining_calls > 10:
                # Nelder-Mead requires at least dim+1 points, but we give it more freedom
                max_iter = max(20, remaining_calls // (dim + 1))
                
                try:
                    # Initialize simplex around best_x with reasonable spread
                    x0 = best_x.copy()
                    
                    result = scipy.optimize.minimize(
                        safe_func, 
                        x0, 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': max_iter,
                            'xatol': 1e-10,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                except (TimeoutError, StopIteration, Exception):
                    pass
        
        # Phase 3: Final Polishing with very small (1+1)-ES
        # Sometimes Nelder-Mead gets stuck in a local minimum or plateaus
        if calls < budget and time.time() < deadline and best_x is not None:
            remaining_calls = budget - calls
            if remaining_calls > 5:
                x = best_x.copy()
                fx = best_f  # No need to re-evaluate; best_f is already the value at best_x
                
                sigma = 0.01
                for _ in range(remaining_calls):
                    if calls >= budget or time.time() > deadline:
                        break
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    try:
                        fy = safe_func(y)
                    except (TimeoutError, StopIteration):
                        break
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.05
                    else:
                        sigma *= 0.95

        return best_x, best_f
