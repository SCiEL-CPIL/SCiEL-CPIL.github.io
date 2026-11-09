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
        # Use 70% of budget for global exploration - leave more room for local refinement
        global_budget = int(budget * 0.70)
        
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
                
            if fy < fx:
                x, fx = y, fy
                # Successful step: increase sigma slightly
                sigma = min(sigma * 1.08, max_sigma)
                stagnation = 0
            else:
                # Failed step: decrease sigma
                sigma = max(sigma * 0.92, min_sigma)
                stagnation += 1
                
                # If stagnating, restart with better strategy
                if stagnation > max_stagnation:
                    # Restart from a diverse point: mix of best and random
                    if best_x is not None:
                        # 40% chance to restart near best, 60% from diverse random
                        if rng.random() < 0.4:
                            # Perturb best with moderate spread
                            x = best_x + 2.0 * rng.standard_normal(dim)
                            x = np.clip(x, -5, 5)
                        else:
                            # Diverse restart: pure random or Latin hypercube-inspired
                            if rng.random() < 0.7:
                                x = rng.uniform(-5, 5, dim)
                            else:
                                # Structured random: use different random seed pattern
                                x = 5.0 * (2.0 * rng.random(dim) - 1.0)
                            x = np.clip(x, -5, 5)
                        try:
                            fx = safe_func(x)
                        except (TimeoutError, StopIteration):
                            break
                    else:
                        x = rng.uniform(-5, 5, dim)
                        try:
                            fx = safe_func(x)
                        except (TimeoutError, StopIteration):
                            break
                    
                    # Reset parameters for fresh start
                    sigma = 3.0
                    stagnation = 0
                    max_stagnation = max(10, max_stagnation // 2)

        # Phase 2: Local Refinement using Nelder-Mead
        # Nelder-Mead is derivative-free and robust for non-smooth functions like Schwefel
        if calls < budget and time.time() < deadline and best_x is not None:
            remaining_calls = budget - calls
            if remaining_calls > 15:
                # Nelder-Mead: use a fraction of remaining budget
                # Each iteration uses ~dim+1 calls, so max_iter should be conservative
                max_iter = min(remaining_calls // (dim + 3), 40)
                max_iter = max(8, max_iter)
                
                try:
                    # Initialize simplex around best_x with reasonable spread
                    x0 = best_x.copy()
                    
                    # Create initial simplex with controlled spread
                    spread = 0.5
                    initial_simplex = [x0]
                    for i in range(dim):
                        point = x0.copy()
                        point[i] += spread
                        initial_simplex.append(point)
                    
                    result = scipy.optimize.minimize(
                        safe_func, 
                        x0, 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': max_iter,
                            'xatol': 1e-6,
                            'fatol': 1e-6,
                            'adaptive': True,
                            'initial_simplex': initial_simplex
                        }
                    )
                except (TimeoutError, StopIteration, Exception):
                    pass
        
        # Phase 3: Final Polishing with adaptive (1+1)-ES
        # Sometimes Nelder-Mead gets stuck in a local minimum or plateaus
        if calls < budget and time.time() < deadline and best_x is not None:
            remaining_calls = budget - calls
            if remaining_calls > 10:
                x = best_x.copy()
                fx = best_f  # best_f is already the value at best_x
                
                # Start with moderate sigma and adapt
                sigma = 0.2
                stagnation = 0
                max_stagnation_polish = 20
                improvement_threshold = 1e-10
                
                for _ in range(remaining_calls):
                    if calls >= budget or time.time() > deadline:
                        break
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    try:
                        fy = safe_func(y)
                    except (TimeoutError, StopIteration):
                        break
                    if fy < fx - improvement_threshold:
                        x, fx = y, fy
                        # Successful step: increase sigma slightly
                        sigma = min(sigma * 1.03, 1.0)
                        stagnation = 0
                    else:
                        # Failed step: decrease sigma
                        sigma = max(sigma * 0.85, 0.005)
                        stagnation += 1
                        if stagnation > max_stagnation_polish:
                            # If stuck, try a slightly larger jump or break
                            if sigma > 0.01:
                                sigma = 0.3
                            else:
                                break
                            stagnation = 0
                            max_stagnation_polish = max(8, max_stagnation_polish // 2)

        return best_x, best_f
