import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290
        rng = np.random.default_rng(0)
        
        # Start near origin since minimizer is in [-4,4]^5
        x_best = np.zeros(dim)
        fx_best = func(x_best)
        calls = 1
        
        # Diagonal covariance adaptation
        sigma = np.ones(dim)
        
        # Track recent step successes for adaptation
        success_count = 0
        window_size = 10
        
        # Track recent directions for correlation
        recent_steps = []
        recent_successes = []
        
        for i in range(1, budget):
            if time.time() > deadline:
                break
            
            # Generate candidate with potential correlation
            if len(recent_steps) >= 5:
                # Use a mix of independent and correlated steps
                if rng.random() < 0.3:
                    # Correlated step based on recent successful directions
                    recent_steps_arr = np.array(recent_steps[-5:])
                    recent_successes_arr = np.array(recent_successes[-5:])
                    # Weight by success (1 for success, 0 for failure)
                    weighted_steps = recent_steps_arr * recent_successes_arr[:, np.newaxis]
                    avg_step = np.sum(weighted_steps, axis=0) / (np.sum(recent_successes_arr) + 1e-8)
                    step = sigma * (0.5 * rng.standard_normal(dim) + 0.5 * avg_step)
                else:
                    step = sigma * rng.standard_normal(dim)
            else:
                step = sigma * rng.standard_normal(dim)
            
            y = np.clip(x_best + step, -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx_best:
                # Success
                x_best = y
                fx_best = fy
                success_count += 1
                recent_steps.append(step)
                recent_successes.append(1)
                
                # Increase sigma for dimensions that improved
                improved_mask = step != 0
                sigma[improved_mask] *= 1.15
            else:
                # Failure
                # Decrease sigma for all dimensions
                sigma *= 0.95
                recent_steps.append(step)
                recent_successes.append(0)
            
            # Keep only last 20 steps
            if len(recent_steps) > 20:
                recent_steps = recent_steps[-20:]
                recent_successes = recent_successes[-20:]
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-4, 5.0)
            
            # Periodic adaptation: if too many successes, increase; if too few, decrease
            if i % window_size == 0:
                if success_count / window_size > 0.5:
                    sigma *= 1.1
                elif success_count / window_size < 0.2:
                    sigma *= 0.8
                success_count = 0
        
        return x_best, fx_best
