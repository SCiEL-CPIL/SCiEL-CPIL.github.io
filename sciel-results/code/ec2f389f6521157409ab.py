import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Use 290s to be safe against the 300s limit
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(42)
        
        # Initial point
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        
        # Adaptive parameters
        sigma = 2.0
        lam = 8
        target_success = 0.2
        
        # Track best
        best_x = x.copy()
        best_fx = fx
        
        calls_made = 1
        
        # Main loop
        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Determine how many offspring we can evaluate
            remaining = self.budget - calls_made
            current_lam = min(lam, remaining)
            
            if current_lam <= 0:
                break
            
            # Generate offspring
            noise = rng.standard_normal((current_lam, self.dim))
            Y = x + sigma * noise
            Y = np.clip(Y, -5, 5)
            
            # Evaluate offspring
            # List comprehension is fast enough for small lam
            F = np.array([func(y) for y in Y])
            calls_made += current_lam
            
            # Check time again
            if time.time() > deadline:
                break
            
            # Find best offspring
            best_idx = np.argmin(F)
            best_y = Y[best_idx]
            best_fy = F[best_idx]
            
            # Update global best
            if best_fy < best_fx:
                best_fx = best_fy
                best_x = best_y.copy()
            
            # Adapt step size
            successes = np.sum(F < fx)
            success_rate = successes / current_lam
            
            if success_rate > target_success:
                sigma *= 1.2
            elif success_rate < target_success:
                sigma *= 0.8
            else:
                sigma *= 1.05
                
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 0.01, 10.0)
            
            # Update current point to the best offspring
            x = best_y
            fx = best_fy
            
        return best_x, best_fx
