"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Linear gradient estimation and corner jump."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, leave 10s margin
        
        dim = self.dim
        budget = self.budget
        
        # 1. Estimate gradient at x=0
        # Use central differences for robustness
        x0 = np.zeros(dim)
        f0 = func(x0)
        
        # Check time
        if time.time() > deadline:
            return x0, f0

        eps = 1.0
        grad = np.zeros(dim)
        
        calls_made = 1
        
        for i in range(dim):
            if time.time() > deadline:
                break
            
            xp = x0.copy()
            xm = x0.copy()
            xp[i] += eps
            xm[i] -= eps
            
            fp = func(xp)
            calls_made += 1
            
            if time.time() > deadline:
                break
                
            fm = func(xm)
            calls_made += 1
            
            grad[i] = (fp - fm) / (2.0 * eps)
            
        # 2. Determine optimal corner
        # If grad[i] > 0, decrease x[i] -> -5
        # If grad[i] < 0, increase x[i] -> 5
        x_opt = np.zeros(dim)
        for i in range(dim):
            if grad[i] > 0:
                x_opt[i] = -5.0
            else:
                x_opt[i] = 5.0
                
        # 3. Evaluate at optimal corner
        if time.time() > deadline:
             # If we ran out of time, return best so far
             # We don't have a best so far other than f0, but x0 is not optimal.
             # We should return the last evaluated point if possible, or x0.
             # Actually, we should just return x0, f0 if we didn't finish.
             # But we might have evaluated some points.
             # Let's just return x0, f0 for safety if we didn't reach x_opt.
             # Or better, if we have budget, we can try to evaluate x_opt if time permits.
             pass
        
        f_opt = func(x_opt)
        calls_made += 1
        
        # 4. Polish with remaining budget if any
        # Since it's linear, x_opt is the global min.
        # We can just use the remaining budget to call func with x_opt 
        # to ensure we hit the 500 call limit if required, or just stop.
        # The problem says "repeating the last n_i if fewer calls are made".
        # So stopping is fine and efficient.
        
        # However, to be safe against "exception before first call" (we made calls)
        # and to potentially improve if the function wasn't perfectly linear (e.g. noise),
        # we could do a few random checks. But for BBOB f5, it is deterministic.
        
        # Let's just return the result.
        return x_opt, f_opt

# EVOLVE-BLOCK-END
