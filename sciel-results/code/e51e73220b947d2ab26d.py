import numpy as np
import time

class Optimizer:
    """
    Optimizer using a (1+1)-ES with stagnation-based restarts.
    Designed for 5D Rastrigin-like functions with 500 function evaluations.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Time guard
        start_time = time.time()
        time_limit = 290.0  # 300s limit, 10s margin
        
        # Initial point: Center of the box is a good guess for Rastrigin
        x = np.zeros(self.dim)
        fx = func(x)
        
        best_x = x.copy()
        best_fx = fx
        
        sigma = 1.0  # Initial step size. Rastrigin local minima are 1 unit apart.
        
        no_improve_count = 0
        stagnation_threshold = 30  # Restart if no improvement for this many steps
        
        calls_made = 1
        
        while calls_made < self.budget:
            # Check time
            if time.time() - start_time > time_limit:
                break
            
            # Generate candidate
            # Use standard normal distribution
            step = rng.standard_normal(self.dim)
            y = x + sigma * step
            
            # Clip to box [-5, 5]
            y = np.clip(y, -5.0, 5.0)
            
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                # Accept step
                x = y
                fx = fy
                
                # Update global best
                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Increase step size on success (1/5 rule variant)
                sigma *= 1.5
            else:
                # Reject step
                no_improve_count += 1
                
                # Decrease step size on failure
                sigma *= 0.87
                
                # Prevent sigma from becoming too small
                if sigma < 1e-6:
                    sigma = 1e-6

            # Stagnation Handling
            if no_improve_count >= stagnation_threshold:
                # We are likely stuck in a local minimum.
                # Strategy: Restart from a new random point or take a large jump.
                
                # 50% chance to jump to a completely new random location
                # 50% chance to stay but increase sigma drastically to try to escape
                if rng.random() < 0.5:
                    # New random start in [-4, 4]
                    x = rng.uniform(-4.0, 4.0, self.dim)
                    fx = func(x)
                    calls_made += 1
                    
                    if fx < best_fx:
                        best_x = x.copy()
                        best_fx = fx
                    
                    sigma = 1.0
                else:
                    # Large jump from current position
                    # Increase sigma significantly
                    sigma *= 3.0
                    
                no_improve_count = 0
        
        return best_x, best_fx
