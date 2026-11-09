import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin
        
        d = self.dim
        budget = self.budget
        
        # Phase 1: Coarse Grid/Random Search to find a good basin
        # Use a portion of the budget for exploration
        # Since the optimum is in [-4, 4]^5, we can bias our search there
        # Let's use 50% of the budget for initial exploration if budget is large enough
        # But we need to be careful about time. Let's do a fixed number of initial points
        # or a fraction of the budget.
        
        # Strategy: 
        # 1. Evaluate a set of random points, biased towards center.
        # 2. Find the best point.
        # 3. Run a local optimizer (like Nelder-Mead or L-BFGS-B) from that point.
        # 4. If time/budget remains, restart from other top points or random points.
        
        # Let's use a simpler approach: 
        # 1. Random search for 50% of budget (or until time limit).
        # 2. Then local refinement from the best point.
        
        # Actually, the directive suggests: "first, use a coarse grid-based random search... second, apply a CMA-ES optimizer".
        # The failed program was CMA-ES. The error was likely due to complexity or bugs in the manual CMA-ES implementation.
        # Let's use scipy.optimize.minimize with Nelder-Mead which is robust and fast for low dimensions.
        
        # To improve AOCC, we want early progress. 
        # Random search provides immediate evaluations.
        # Local search refines.
        
        # Let's split the budget:
        # 50% for global exploration (random)
        # 50% for local exploitation (Nelder-Mead from best global point)
        
        n_global = budget // 2
        n_local = budget - n_global
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        # Helper to evaluate and track
        def evaluate(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Phase 1: Global Search
        # Generate random points. To ensure coverage, we can mix uniform in [-4,4]^5 and [-5,5]^5
        # Since min is in [-4,4]^5, biasing there is good.
        
        # We will generate points in batches to check time
        batch_size = 100
        
        while calls < n_global and time.time() < deadline:
            # Determine how many points we can afford in this batch
            # Just generate batch_size points
            # But check time before each func call? No, that's too slow.
            # Check time before the batch.
            
            # Generate random points
            # Mix of uniform [-4,4] and [-5,5]
            # 80% in [-4,4], 20% in [-5,5]
            n_pts = min(batch_size, n_global - calls)
            if n_pts <= 0:
                break
            
            # Generate n_pts points
            X = np.random.uniform(-4.0, 4.0, size=(n_pts, d))
            # Add some points from full range to be safe
            # Let's just use [-4,4] for most, and maybe a few from [-5,5]
            # Actually, let's just use [-4,4] for the main exploration as per hint
            # And maybe add a few random points from [-5,5]
            
            # Let's mix: 70% in [-4,4], 30% in [-5,5]
            n_inner = int(n_pts * 0.7)
            n_outer = n_pts - n_inner
            
            X_inner = np.random.uniform(-4.0, 4.0, size=(n_inner, d))
            X_outer = np.random.uniform(-5.0, 5.0, size=(n_outer, d))
            X = np.vstack([X_inner, X_outer])
            
            for x in X:
                if calls >= n_global or time.time() > deadline:
                    break
                evaluate(x)
                
        # Phase 2: Local Refinement
        # Use the best point found in Phase 1
        if best_x is not None:
            # Use Nelder-Mead
            # We need to be careful with the number of function calls in minimize
            # Nelder-Mead in d=5 might use many calls.
            # We have n_local budget left.
            
            # To control calls, we can use a wrapper that counts and stops if budget exceeded
            # But scipy.optimize.minimize doesn't easily stop mid-iteration based on a counter without exceptions or custom callbacks (which are limited).
            # We can just run it and hope it finishes within budget/time.
            # Given d=5, Nelder-Mead is fast.
            
            # Let's run multiple local searches from the top few points found in Phase 1?
            # The current code only keeps the best.
            # Let's keep track of the top 3 points from Phase 1.
            
            # Re-implementing Phase 1 to keep top points would be better, but let's stick to the current structure for minimal changes first.
            # Actually, let's just run Nelder-Mead from the best point.
            
            # We need to ensure we don't exceed the total budget.
            # The minimize function will call func.
            # We can pass a wrapper that checks the budget.
            
            local_calls = 0
            def local_func(x):
                nonlocal local_calls, calls, best_x, best_f
                if calls >= budget or time.time() > deadline:
                    return best_f # Return a high value or the best found to stop? 
                                 # Returning best_f might cause convergence issues.
                                 # Better to raise an exception or return a very high value.
                                 # But the contract says "exception later keeps the calls made".
                                 # So an exception is safe for the score, but might kill the program?
                                 # "an exception later keeps the calls made" implies the program continues or ends?
                                 # "More than 500 calls... makes the whole result invalid"
                                 # So we must NOT exceed 500 calls.
                                 # If we exceed, the result is invalid.
                                 # So we must stop calling func if calls >= budget.
                                 
                if calls >= budget:
                    return 1e9 # High value to discourage further moves
                local_calls += 1
                calls += 1
                x = np.clip(x, -5.0, 5.0)
                f = func(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            
            try:
                # Use Nelder-Mead
                # maxiter and maxfev
                # We have n_local calls remaining roughly.
                # Set maxfev to n_local to be safe.
                res = minimize(local_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': n_local, 'maxfev': n_local, 'xatol': 1e-4, 'fatol': 1e-6})
            except Exception:
                pass
            
        return best_x, best_f
