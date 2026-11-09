"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start (1+1)-ES optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill
        
        dim = self.dim
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility in exploration
        
        # Phase 1: Diverse Initialization
        # Use a portion of the budget to sample random points and find a good starting basin.
        # For dim=5, 50 initial samples is cheap and helps avoid poor local minima.
        n_init = min(50, self.budget // 6)
        
        best_x = None
        best_f = np.inf
        
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            if time.time() - start_time > time_limit:
                return best_x, best_f
        
        # Phase 2: (1+1)-ES
        # Parameters for (1+1)-ES
        # Standard (1+1)-ES with step size adaptation based on 1/5 success rule
        sigma = 2.0  # Initial step size
        c_sigma = 1.0 / (dim + 4)  # Learning rate for sigma
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        x = best_x
        fx = best_f
        
        remaining_budget = self.budget - n_init
        
        while remaining_budget > 0:
            if time.time() - start_time > time_limit:
                break
            
            # Generate offspring
            y = rng.normal(0, 1, dim)
            x_new = x + sigma * y
            x_new = np.clip(x_new, -5, 5)
            
            fx_new = func(x_new)
            remaining_budget -= 1
            
            # Update best
            if fx_new < best_f:
                best_f = fx_new
                best_x = x_new.copy()
            
            # Step size adaptation (1/5 success rule)
            # If improvement, increase sigma slightly; otherwise decrease.
            # Standard update: sigma *= exp(c_sigma * (I[improved] - 0.2))
            # But a common variant is: sigma *= exp(c_sigma * (success - 0.2))
            # Where success is 1 if improved, 0 otherwise.
            # To be more aggressive, we can use a larger c_sigma or different target.
            # Let's use the standard 1/5 rule but with a slightly higher target for exploration early on?
            # No, standard is robust.
            
            success = 1.0 if fx_new < fx else 0.0
            sigma *= np.exp(c_sigma * (success - 0.2))
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Move to offspring if it's better, or with some probability if it's worse?
            # Standard (1+1)-ES only moves if better.
            if fx_new < fx:
                x = x_new
                fx = fx_new
            
            # If sigma becomes very small, we might be stuck.
            # A simple restart mechanism: if no improvement in last 50 steps, restart from best_x with larger sigma?
            # For now, keep it simple. The diverse init should have found a good basin.
            
            # Optional: If we are stagnating (sigma very small and no progress), we could do a local random restart.
            # But budget is tight. Let's stick to the main loop.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
