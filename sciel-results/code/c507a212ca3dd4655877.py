"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        # We will use multiple starts to cover the basin of attraction
        # 10 starts, 50 evaluations each = 500 total
        n_starts = 10
        budget_per_start = self.budget // n_starts
        
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = np.inf
        
        for i in range(n_starts):
            if time.time() > deadline:
                break
            
            # Generate diverse initial points
            # Use uniform distribution in [-5, 5]
            x0 = rng.uniform(-5.0, 5.0, self.dim)
            
            # Track local best for this start
            local_best_f = np.inf
            local_best_x = x0
            
            # Define a wrapper to count evaluations and update global best
            # SciPy's minimize calls the function many times, so we need to be careful
            # with the budget. We will pass a maxiter that roughly corresponds to the budget.
            # Note: Nelder-Mead iterations consume variable numbers of function calls.
            # A safe maxiter is usually smaller than the budget.
            # Let's estimate: 1 Nelder-Mead iteration ~ 5-10 function calls.
            # So maxiter ~ budget_per_start / 5
            max_iter = max(1, budget_per_start // 5)
            
            def objective(x):
                nonlocal local_best_f, local_best_x, best_f, best_x
                # Clip to box as per task description (though func does it, good to be safe)
                x_clipped = np.clip(x, -5.0, 5.0)
                fx = func(x_clipped)
                
                if fx < local_best_f:
                    local_best_f = fx
                    local_best_x = x_clipped.copy()
                
                if fx < best_f:
                    best_f = fx
                    best_x = x_clipped.copy()
                    
                return fx

            try:
                # Run Nelder-Mead
                # options: maxiter controls the number of iterations
                # We don't pass 'maxfev' directly in all versions easily for strict counting,
                # but maxiter is a good proxy.
                res = minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                
                # Update best with the result if it's better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                # If minimize fails, we still have the local_best from the wrapper calls
                pass

        # Return the best found
        if best_x is None:
            # Fallback if no evaluations were made (should not happen)
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
