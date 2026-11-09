import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Time management
        start_time = time.time()
        # 330s limit, use 325s as hard stop to be safe against overhead.
        time_limit = 325.0 

        x_best = None
        f_best = np.inf
        eval_count = 0
        
        # Phase 1: Multi-start initialization
        # Evaluate 10 random points + center to find a good starting basin
        init_points = np.zeros((1, n))
        init_points = np.vstack([init_points, np.random.uniform(-5, 5, (9, n))])
        
        for i in range(10):
            if time.time() - start_time > time_limit:
                break
            x = init_points[i]
            fx = func(x)
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            eval_count += 1
            
        if x_best is None:
            # Fallback if something went wrong (shouldn't happen)
            x_best = np.zeros(n)
            f_best = func(x_best)
            eval_count = 1

        # Phase 2: (1+1)-ES with adaptive sigma and restarts
        # We use the best point from init as the starting point
        x = x_best.copy()
        fx = f_best
        
        sigma = 1.5  # Initial step size, slightly smaller than 2.0 for better convergence
        sigma_min = 1e-10
        sigma_max = 8.0
        
        # Parameters for 1/5 rule approximation
        # We use a simple multiplicative update:
        # Success: sigma *= 1.05
        # Failure: sigma *= 0.95
        # This is a gentler adaptation than 1.2/0.85, which helps in noisy or complex landscapes
        
        steps_since_best_improvement = 0
        restart_threshold = 40  # More patient before restart to allow for slower progress
        
        # Track multiple local optima for diversity
        # Store (point, value) pairs
        local_optima = [(x_best.copy(), f_best)]
        
        # Keep the best 5 local optima
        max_local_optima = 5
        
        while eval_count < budget:
            if time.time() - start_time > time_limit:
                break
            
            # Generate candidate
            z = np.random.standard_normal(n)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            
            try:
                fy = func(y)
            except Exception:
                break
                
            eval_count += 1
            
            if fy < fx:
                # Success
                x = y
                fx = fy
                sigma *= 1.05
                
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
                    steps_since_best_improvement = 0
                    # Update local optima: add the new best
                    local_optima.append((x_best.copy(), f_best))
                    # Sort by value and keep only the best 5
                    local_optima.sort(key=lambda t: t[1])
                    local_optima = local_optima[:max_local_optima]
                else:
                    steps_since_best_improvement += 1
            else:
                # Failure
                sigma *= 0.95
                steps_since_best_improvement += 1
                
            # Clamp sigma
            if sigma > sigma_max:
                sigma = sigma_max
            if sigma < sigma_min:
                sigma = sigma_min
                
            # Restart mechanism
            # If we haven't improved the global best in 'restart_threshold' steps,
            # reset the local search to a random local optimum or the best point
            if steps_since_best_improvement >= restart_threshold:
                # Pick a random local optimum or the best one
                if len(local_optima) > 1:
                    # With 30% probability, pick a non-best local optimum to explore a different basin
                    if np.random.random() < 0.3:
                        idx = np.random.randint(1, len(local_optima))
                    else:
                        idx = 0
                    x = local_optima[idx][0].copy()
                    fx = local_optima[idx][1]
                else:
                    x = x_best.copy()
                    fx = f_best
                
                # Reset sigma to a reasonable exploration size
                sigma = 2.0
                steps_since_best_improvement = 0
                
                # Occasionally add a new random point to local_optima for diversity
                if np.random.random() < 0.2 and len(local_optima) < max_local_optima:
                    x_new = np.random.uniform(-5, 5, n)
                    try:
                        f_new = func(x_new)
                        eval_count += 1
                        local_optima.append((x_new.copy(), f_new))
                        # Sort by value and keep only the best 5
                        local_optima.sort(key=lambda t: t[1])
                        local_optima = local_optima[:max_local_optima]
                    except Exception:
                        pass
        
        # Phase 3: Final Polish
        # Use remaining budget for high-precision local search from the best point
        if eval_count < budget and time.time() - start_time < time_limit:
            x = x_best.copy()
            fx = f_best
            sigma = 0.1  # Small sigma for fine-tuning
            
            max_polish_steps = min(budget - eval_count, 50)
            
            for _ in range(max_polish_steps):
                if eval_count >= budget or time.time() - start_time > time_limit:
                    break
                
                z = np.random.standard_normal(n)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                
                try:
                    fy = func(y)
                except Exception:
                    break
                    
                eval_count += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fy < f_best:
                        f_best = fy
                        x_best = y.copy()
                    sigma *= 0.95  # Slightly decrease sigma on success
                else:
                    sigma *= 0.9  # Decrease sigma on failure for faster convergence
                
                sigma = np.clip(sigma, 1e-12, 0.5)
        
        return x_best, f_best
