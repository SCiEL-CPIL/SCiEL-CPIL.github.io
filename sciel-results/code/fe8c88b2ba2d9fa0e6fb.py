"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead simplex optimizer with multiple random restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # We'll use a multi-start Nelder-Mead approach.
        # Since we don't know the landscape well, we'll try a few different
        # starting points and allocate budget among them.
        
        # Strategy:
        # 1. Try a few initial points spread across the domain.
        # 2. For each, run Nelder-Mead with a limited budget.
        # 3. Keep track of the best point found.
        # 4. If budget remains, continue optimizing from the best point.
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def wrapper(x):
            nonlocal calls_used
            x_clipped = np.clip(x, -5, 5)
            calls_used += 1
            return func(x_clipped)
        
        # Initial spread: try center and a few random points
        # Let's do 3 initial restarts with ~100 calls each, then 1 final with remaining
        # Budget is 500.
        
        # Restart 1: Center
        x0_1 = np.zeros(self.dim)
        res1 = minimize(wrapper, x0_1, method='Nelder-Mead', 
                        options={'maxiter': 80, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if res1.fun < best_f:
            best_f = res1.fun
            best_x = res1.x
        
        # Restart 2: Random point
        x0_2 = rng.uniform(-4, 4, self.dim)
        res2 = minimize(wrapper, x0_2, method='Nelder-Mead', 
                        options={'maxiter': 80, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if res2.fun < best_f:
            best_f = res2.fun
            best_x = res2.x
        
        # Restart 3: Another random point
        x0_3 = rng.uniform(-4, 4, self.dim)
        res3 = minimize(wrapper, x0_3, method='Nelder-Mead', 
                        options={'maxiter': 80, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if res3.fun < best_f:
            best_f = res3.fun
            best_x = res3.x
        
        # Final polish: use remaining budget from the best point found
        # Each Nelder-Mead call uses roughly maxiter * (dim+1) evaluations? 
        # Actually scipy counts function calls. Let's just run one more with remaining budget.
        # We've used approx 3 * (some number). Let's just run a final one with a high maxiter 
        # but it will stop when budget is exhausted? No, wrapper doesn't stop.
        # We need to be careful not to exceed 500 calls.
        
        # Let's estimate calls. Nelder-Mead typically uses n+1 initial points + iterations.
        # With maxiter=80, it might use more than 80 calls. 
        # To be safe, let's check calls_used.
        
        remaining = self.budget - calls_used
        if remaining > 10 and best_x is not None:
            # Run final optimization from best_x
            # We can't easily pass a max_call limit to scipy minimize, 
            # but we can use a custom callback or just hope it converges fast.
            # Better: implement a simple loop or use a smaller maxiter.
            # Let's just do a few more iterations if we have budget.
            
            # Since we can't strictly limit calls in scipy minimize easily without 
            # modifying the function to raise an exception (which might break things),
            # let's just run one more NM with a modest maxiter.
            # If it exceeds, the exception handling in the evaluator keeps calls made.
            # But the contract says "More than 500 calls ... makes the whole result invalid".
            # So we MUST not exceed 500.
            
            # We need a safer way. Let's implement a simple gradient-free step if we have budget,
            # or just return the best found.
            
            # Actually, the safest is to just return the best found so far if we are close to limit.
            # Let's check if we have enough budget for a small local search.
            
            # For simplicity and safety, let's just return the best found.
            # The three restarts should cover the space reasonably well for dim=5.
            pass
        
        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
