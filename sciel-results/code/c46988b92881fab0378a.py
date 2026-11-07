"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-start L-BFGS-B for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s kill

        # Count calls to respect budget
        call_count = 0
        max_calls = self.budget

        # Wrap func to count calls and clip
        def counted_func(x):
            nonlocal call_count
            if call_count >= max_calls:
                # Return a large value to stop optimization if budget exceeded
                return 1e10
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        # Define starting points: a grid-based approach
        # Katsuura function has many local minima. A grid of points helps find the global basin.
        # dim=5, box [-5, 5]^5.
        # We'll use a small grid plus some random points.
        # To maximize coverage within budget, we need to be efficient.
        # L-BFGS-B typically takes ~10-50 function evaluations to converge in 5D.
        # With 500 budget, we can afford ~10-20 restarts.
        
        # Let's use a deterministic set of starting points.
        # 1. Center
        # 2. Corners (2^5 = 32, too many? maybe sample some)
        # 3. Grid points
        
        # Strategy: Use a fixed set of starting points that cover the space.
        # Let's try a 2-point grid per dimension (low, high, mid) -> 3^5 = 243 points. Too many for 500 calls if each takes 10+ calls.
        # Let's try 2 points per dimension: 2^5 = 32 points.
        # 500 / 32 = ~15 calls per point. Might not converge.
        
        # Better: Randomized multi-start with a fixed seed for reproducibility, but adaptive.
        # Or: A curated list of starting points.
        
        # Let's use a mix:
        # 1. Origin (0,0,0,0,0)
        # 2. Random points from a fixed RNG
        
        rng = np.random.default_rng(42)
        
        # Generate a list of starting points
        # We will try them in order of expected promise or just sequentially.
        # To maximize AOCC, we want to get good values early.
        # L-BFGS-B from a random point might take a few steps to get good.
        
        # Let's create a list of candidate starts.
        starts = []
        # Center
        starts.append(np.zeros(self.dim))
        
        # Add random points
        for _ in range(20):
            starts.append(rng.uniform(-4, 4, self.dim))
            
        # Sort starts? No, just iterate.
        # We will run L-BFGS-B from each start, but stop if we run out of budget or time.
        
        best_x = None
        best_f = np.inf
        
        for x0 in starts:
            # Check time and budget
            if time.time() > deadline - 10: # 10s margin for final check
                break
            if call_count >= max_calls - 10: # Leave a few calls for safety
                break
                
            try:
                # Use L-BFGS-B
                # maxiter limits the number of iterations, but we also have call budget.
                # We can't easily limit function calls inside scipy.optimize without wrapping.
                # Our counted_func handles the call limit by returning 1e10.
                # However, if it returns 1e10, L-BFGS-B might keep going.
                # It's better to break out if we hit the limit.
                
                # Let's wrap it more carefully to raise an exception or signal stop.
                # Actually, if counted_func returns 1e10, the optimizer will likely think it's diverging or just get stuck.
                # A better way: check call_count inside the loop of starts.
                
                # Since we can't interrupt scipy.optimize easily, we rely on the budget check in the outer loop.
                # If we run out of calls mid-optimization, the remaining starts won't run.
                
                res = scipy.optimize.minimize(
                    counted_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'maxfun': max_calls - call_count}
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                    
            except Exception:
                pass
        
        # If no valid result found (shouldn't happen), fallback to a single point
        if best_x is None:
            x = np.zeros(self.dim)
            fx = counted_func(x)
            best_x = x
            best_f = fx

        return best_x, best_f

# EVOLVE-BLOCK-END
