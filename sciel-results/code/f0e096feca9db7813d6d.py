"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+lambda)-ES with adaptive step size, stagnation restart, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Initial point
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        # Adaptive step size parameters
        sigma = 2.0  # Start larger to explore basin
        lambda_ = 8  # number of offspring per iteration
        
        # Target success rate for adaptation
        # With lambda=8, target ~1 success per gen is reasonable for exploration
        target_success = 0.2 
        c_sigma = 0.5  # Adaptation speed
        
        calls_made = 1
        no_improve_count = 0
        stagnation_threshold = 40  # Calls without improvement before restart

        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate offspring
            improvements = 0
            best_y = None
            best_fy = fx
            
            for j in range(lambda_):
                # Check time periodically
                if time.time() > deadline:
                    break
                if calls_made >= self.budget:
                    break
                    
                step = sigma * rng.standard_normal(self.dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < best_fy:
                    best_fy = fy
                    best_y = y
                    improvements += 1
            
            # Adapt step size based on success rate
            actual_success = improvements / lambda_
            
            # Adjust sigma
            if actual_success > target_success:
                sigma *= (1 + c_sigma * (actual_success - target_success))
            else:
                sigma *= (1 - c_sigma * (target_success - actual_success))
            
            # Clamp sigma
            sigma = np.clip(sigma, 0.01, 10.0)
            
            # Accept the best offspring
            if best_y is not None:
                x = best_y
                fx = best_fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            # Stagnation Restart
            # If we haven't improved the global best for a while, restart locally
            if no_improve_count >= stagnation_threshold:
                # Restart from a random point near the current best or fully random?
                # Fully random is safer for global search, but might be too disruptive.
                # Let's restart from a point perturbed from the global best with large sigma
                # Or just pick a new random point if we have budget.
                if calls_made < self.budget - 20: # Need some budget left
                    # Pick a new random starting point
                    x = rng.uniform(-5, 5, self.dim)
                    fx = func(x)
                    calls_made += 1
                    # Update best if this random point is better (unlikely but possible)
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    
                    # Reset local search parameters
                    sigma = 2.0
                    no_improve_count = 0
                    stagnation_threshold = 30 # Lower threshold for subsequent restarts

        return best_x, best_fx
# EVOLVE-BLOCK-END
