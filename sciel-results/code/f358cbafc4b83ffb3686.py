"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Restarting (1+1)-ES with 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Global best tracking
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        calls_made = 1
        
        # Parameters for restart strategy
        # We allow up to 'max_restarts' restarts or until budget is exhausted.
        # Stagnation threshold: if no improvement for this many steps, restart.
        stagnation_threshold = 20
        
        # Reserve some budget for final polish
        polish_budget = 80
        es_budget = self.budget - polish_budget
        
        while calls_made < es_budget:
            # Check time limit
            if time.time() > deadline:
                break
            
            # Initialize current run state
            x = x_best.copy()
            # Start with a moderate sigma, maybe slightly larger for exploration
            sigma = 1.5 
            stagnation_count = 0
            local_best_val = f_best
            local_best_x = x_best.copy()
            
            # Run (1+1)-ES for this restart phase
            while calls_made < es_budget:
                if time.time() > deadline:
                    break
                
                # Generate neighbor
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < local_best_val:
                    # Improvement
                    local_best_val = fy
                    local_best_x = y
                    x = y
                    sigma *= 1.2
                    stagnation_count = 0
                    
                    # Update global best
                    if fy < f_best:
                        f_best = fy
                        x_best = y
                else:
                    # No improvement
                    sigma *= 0.9
                    stagnation_count += 1
                
                # Check for stagnation to trigger restart
                if stagnation_count >= stagnation_threshold:
                    break
        
            # If we broke out due to stagnation, the next loop iteration 
            # will restart from x_best with a new sigma.
            # If we broke out due to budget/time, the outer loop condition handles it.
            
            # Safety: If sigma becomes too small, it might be worth restarting 
            # even if not fully stagnant? The 1/5 rule adapts sigma. 
            # If sigma < 1e-6, we are likely stuck.
            if sigma < 1e-6 and calls_made < es_budget:
                break
        
        # Final polish pass using L-BFGS-B if we have budget and time left
        if calls_made < self.budget and time.time() < deadline:
            try:
                # Use L-BFGS-B with bounds to polish the solution
                remaining_budget = self.budget - calls_made
                result = minimize(
                    func,
                    x_best,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'maxfun': remaining_budget, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                calls_made += result.nfev
                
                if result.fun < f_best:
                    f_best = result.fun
                    x_best = result.x
            except Exception:
                pass  # If polish fails, keep the ES result

        return x_best, f_best
# EVOLVE-BLOCK-END
