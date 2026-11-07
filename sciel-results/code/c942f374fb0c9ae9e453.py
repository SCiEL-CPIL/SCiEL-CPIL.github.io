"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        
        # 1. Random sampling phase to find good basins
        best_x = None
        best_f = float('inf')
        calls = 0
        
        # Sample 100 points randomly
        n_samples = min(100, self.budget // 5)
        for _ in range(n_samples):
            if time.time() > deadline:
                break
            if calls >= self.budget:
                break
                
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # 2. Local refinement phase using L-BFGS-B
        # We will attempt to refine the best points found.
        # To manage the budget strictly, we track calls via a wrapper.
        # However, the task says "Every call is recorded". 
        # The evaluator likely wraps func or counts calls to the original func.
        # Wait, the task says: "For each instance... evaluator creates Optimizer... and calls it once."
        # "Every call is recorded". This implies the evaluator passes a func that counts calls.
        # So I must ensure I do not exceed 500 calls to `func`.
        
        # Let's refine the top few best points from the sampling phase.
        # Since we only kept the single best in the loop above, let's keep a small history or just refine the best.
        # Actually, for BBOB f4 (Büchi-Rastrigin), multiple local minima are significant.
        # Let's keep a few best candidates.
        
        # Re-do sampling keeping top 5
        candidates = []
        for _ in range(n_samples):
            if time.time() > deadline or calls >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls += 1
            candidates.append((fx, x))
            
        candidates.sort(key=lambda t: t[0])
        
        # Refine top 5 candidates
        num_refine = min(5, len(candidates))
        
        for i in range(num_refine):
            if time.time() > deadline or calls >= self.budget:
                break
                
            fx0, x0 = candidates[i]
            
            # Define a wrapper for scipy to count calls?
            # No, we just need to be careful.
            # L-BFGS-B uses finite differences. In 5D, it might take ~10-20 calls per step.
            # Max iterations should be small to stay within budget.
            
            # We don't have a strict counter inside the scipy call unless we wrap func.
            # But we can't easily wrap func if the evaluator's counter is on the original object.
            # Actually, we CAN wrap it.
            # Let's create a local counter for this specific optimization run to stop if we exceed remaining budget.
            
            remaining_budget = self.budget - calls
            
            # Estimate calls needed. L-BFGS-B in 5D usually takes ~2-3 calls per dimension per iteration?
            # It's safer to limit iterations.
            max_iter = min(50, remaining_budget // 10) # Rough estimate
            
            if max_iter < 5:
                continue
                
            # Use a wrapper to track calls for this specific minimize call
            local_calls = [0]
            def wrapped_func(x):
                if local_calls[0] >= remaining_budget:
                    return best_f # Return current best to stop improvement? Or raise?
                    # Better to just return a large value to discourage further search in this branch?
                    # Or just let it run and check after?
                    # If it exceeds global budget, the result is invalid.
                    # So we MUST stop before exceeding global budget.
                    # Since we don't know exactly how many calls L-BFGS-B will make,
                    # we should use a very conservative max_iter.
                    # Or, we can implement our own simple gradient descent?
                    # No, let's stick to scipy but be very careful.
                    
                    # Actually, if we exceed budget, the whole result is invalid.
                    # So we must guarantee calls < 500.
                    
                    # Let's check time and budget before starting.
                    # We'll set max_iter low.
                    
                    local_calls[0] += 1
                    return func(x)
                    
            try:
                res = minimize(wrapped_func, x0, method='L-BFGS-B', bounds=[(-5, 5)] * self.dim, 
                               options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
                calls += local_calls[0]
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
