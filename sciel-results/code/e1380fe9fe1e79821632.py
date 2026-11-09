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
        deadline = start_time + 290  # Safety margin before 300s limit
        
        # We will use a sequence of L-BFGS-B runs, which is fast and effective for smooth functions.
        # Since the function is black-box and we don't have gradients, we rely on finite differences 
        # inside L-BFGS-B, which uses ~dim evaluations per step.
        # To handle local minima, we use multi-start.
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        
        # Estimate cost of one L-BFGS-B run.
        # L-BFGS-B with finite differences typically takes ~10-20 * dim function evaluations to converge well.
        # Let's assume ~50 evaluations per run on average for dim=5.
        # Budget is 500. So we can afford ~10 runs.
        
        rng = np.random.default_rng(42)
        
        # We'll do a mix:
        # 1. A few random starts with L-BFGS-B
        # 2. If time allows, maybe try a global method? No, keep it simple and fast.
        
        # Bounds
        bounds = [(-5, 5)] * self.dim
        
        # Number of restarts to attempt
        # We'll adapt the number of restarts based on remaining budget/time.
        
        # First, do a quick scan to find a decent starting point?
        # Or just start with a few random points.
        
        # Let's do 5-10 restarts.
        # Each restart:
        # 1. Pick a random x0
        # 2. Run L-BFGS-B with a maxiter limit to ensure it doesn't take too long
        # 3. Update best
        
        # To be safe with the 500 call limit, we need to count calls.
        # L-BFGS-B doesn't easily expose the exact call count in a simple wrapper without hooks.
        # However, we can wrap the function to count calls.
        
        call_count = 0
        def counted_func(x):
            nonlocal call_count
            call_count += 1
            return func(x)
        
        # We'll try a few restarts.
        # Let's start with 10 restarts. If we run out of budget, we stop.
        
        # To prevent L-BFGS-B from using too many evaluations, we set maxiter.
        # For dim=5, maxiter=50 is usually enough for a good local optimum.
        # Each iteration of L-BFGS-B uses ~dim function evaluations for gradient.
        # So 50 * 5 = 250 evaluations per run. That's too many for 10 runs.
        # We need to be more conservative.
        
        # Let's use maxiter=20. 20 * 5 = 100 evaluations per run.
        # 5 runs = 500 evaluations.
        
        # Alternatively, we can use Nelder-Mead which is derivative-free but slower per step.
        # L-BFGS-B with finite differences is generally faster for smooth functions like Rosenbrock.
        
        # Let's try 5 restarts with maxiter=20.
        
        for i in range(10):
            if time.time() > deadline:
                break
            if call_count > self.budget - 10: # Keep a small buffer
                break
                
            # Random start
            x0 = rng.uniform(-5, 5, self.dim)
            
            # Run L-BFGS-B
            try:
                res = minimize(counted_func, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 20, 'ftol': 1e-8, 'gtol': 1e-6})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
                
        return best_x, best_f

# EVOLVE-BLOCK-END
