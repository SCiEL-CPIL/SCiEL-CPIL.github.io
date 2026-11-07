import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin against 330s limit
        
        def time_check():
            return (time.time() - start_time) > timeout

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        # Strategy: Multi-start (1+1)-ES
        # 1. Use 10% of budget for global exploration (multi-start)
        # 2. Use remaining 90% for local refinement with adaptive step size
        
        n_starts = 10
        budget_per_start = max(1, budget // n_starts)
        
        # Phase 1: Multi-start initialization
        # We try 10 random points. For each, we do a small number of (1+1) steps
        # to find a local minimum. We keep the best one.
        
        # Initial step size for exploration
        sigma = 2.0
        
        for s in range(n_starts):
            if eval_count >= budget or time_check():
                break
            
            # Random start in [-4, 4]^5
            x = rng.uniform(-4.0, 4.0, dim)
            f = func(x)
            eval_count += 1
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            
            # Small local search from this point to see if it's promising
            # Use a very small budget for this phase to ensure we get to the main phase
            local_budget = min(budget_per_start // 2, 10) # Max 10 steps per start in phase 1
            local_sigma = 1.0
            
            for _ in range(local_budget):
                if eval_count >= budget or time_check():
                    break
                
                z = rng.standard_normal(dim)
                x_new = x + local_sigma * z
                f_new = func(x_new)
                eval_count += 1
                
                if f_new < f:
                    x = x_new
                    f = f_new
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    # Slightly increase sigma if successful? Or keep constant?
                    # For (1+1)-ES, usually constant or adaptive. Let's keep simple.
                else:
                    # Halve sigma if failed
                    local_sigma *= 0.5
                    
            # If we found a very good point, we might break early?
            # No, let's use the full multi-start budget to be safe.
            # But if we are close to 0, we can stop.
            if best_f < 1e-6:
                break

        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(dim)
            if eval_count < budget:
                best_f = func(best_x)
                eval_count += 1
            else:
                best_f = np.inf

        # Phase 2: Refinement from the best point found
        # Use a (1+1)-ES with adaptive step size
        # Start with a moderate sigma
        sigma = 0.5
        x = best_x.copy()
        f = best_f
        
        # We have remaining budget
        # Adaptive (1+1)-ES: 
        # If new point is better, increase sigma slightly (or keep)
        # If worse, decrease sigma
        # Standard (1+1)-ES rule: sigma *= exp(c * (1 if success else -1)) or similar
        # Let's use a simple heuristic:
        # sigma *= 1.1 if success, sigma *= 0.9 if failure
        # Or better: sigma *= exp(0.1 * (1 if success else -1))
        
        c_sigma = 0.1
        
        while eval_count < budget and not time_check():
            z = rng.standard_normal(dim)
            x_new = x + sigma * z
            f_new = func(x_new)
            eval_count += 1
            
            if f_new < f:
                x = x_new
                f = f_new
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                sigma *= np.exp(c_sigma)
            else:
                sigma *= np.exp(-c_sigma)
                
            # Stop if sigma is too small
            if sigma < 1e-10:
                break
                
        return best_x, best_f
