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
        d = self.dim
        budget = self.budget
        
        calls = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def safe_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5.0, 5.0)
            calls += 1
            try:
                f = float(func(x_clipped))
            except Exception:
                return np.inf
            if not np.isfinite(f):
                return np.inf
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Phase 1: Multi-start BFGS
        # We use a deterministic stratified grid to ensure coverage
        # Let's pick around 10-15 starting points
        # A simple approach: generate points from a Latin Hypercube or just random points
        # Given d=5, let's use 12 starting points
        
        num_starts = 12
        starts = []
        
        # Center point
        starts.append(np.zeros(d))
        
        # Add random points from LHS-like distribution
        # We can just use rng.random for simplicity but stratified is better
        # Let's use a simple stratified approach: divide each dimension into 2-3 bins
        # Or just use random points with a fixed seed for reproducibility
        
        for i in range(num_starts - 1):
            # Sample from [-4.5, 4.5] to stay well inside the box
            p = rng.uniform(-4.5, 4.5, size=d)
            starts.append(p)

        # Run BFGS from each start
        for start in starts:
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Check if we are close to deadline, BFGS might take many iterations
            # We limit the number of function evaluations per BFGS run
            # BFGS usually converges in < 100 iterations for 5D
            # We'll let scipy handle it but check time inside the callback?
            # Scipy minimize doesn't support easy time-based abort in the callback easily
            # without breaking the loop.
            # Instead, we can set a maxfun limit.
            # If we have 500 calls and 12 starts, we have ~40 calls per start.
            # This might be too low for BFGS to converge fully from a bad start.
            # Let's allocate budget dynamically.
            
            remaining = budget - calls
            if remaining <= 0:
                break
            
            # Allocate up to 30% of remaining budget or 50 calls, whichever is smaller
            # But ensure we don't exceed total budget
            max_fun_for_this = min(50, remaining)
            
            try:
                # BFGS uses numerical gradients, which costs 2*d+1 = 11 calls per iteration
                # So 50 calls is about 4-5 iterations. This might be too few.
                # Let's try a different strategy:
                # Use a coarser global search first?
                
                # Actually, for f19, BFGS is very efficient.
                # Let's try allocating more.
                # If we have 500 calls, and we do 5 BFGS runs, we have 100 calls each.
                # 100 calls is ~9 iterations. Still small.
                
                # Alternative: Use Nelder-Mead? It's slower per iteration but robust.
                # Or use COBYLA?
                
                # Let's stick to BFGS but reduce the number of starts to allow deeper search.
                # 5 starts with 100 calls each.
                
                # Let's refine the start selection.
                # We already picked 12. Let's just run them with a small budget first?
                # No, if we run with small budget, we might not find the valley.
                
                # Better: Run BFGS with a generous maxfun, but break if time is low.
                # We can't easily break scipy minimize mid-run without a callback hack.
                
                # Let's use a wrapper that checks time?
                # No, let's just trust the budget.
                
                # Strategy: 
                # 1. Evaluate the start point itself.
                f_start = safe_eval(start)
                if calls >= budget:
                    break
                
                # 2. Run BFGS
                # maxfun: allow up to 80% of remaining budget
                maxfun = int((budget - calls) * 0.8)
                if maxfun < 10:
                    maxfun = 10
                    
                res = minimize(
                    safe_eval, 
                    start, 
                    method='BFGS', 
                    options={'maxiter': 100, 'maxfun': maxfun, 'gtol': 1e-8}
                )
                
            except Exception:
                pass

            if time.time() > deadline:
                break

        # Phase 2: Polish the best point found with a very tight local search
        if best_x is not None and calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining > 10:
                try:
                    # Use Nelder-Mead with tight tolerance for final polish
                    # Or just run BFGS again from best_x
                    maxfun = remaining
                    res = minimize(
                        safe_eval, 
                        best_x, 
                        method='BFGS', 
                        options={'maxiter': 50, 'maxfun': maxfun, 'gtol': 1e-12}
                    )
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(d)
            
        return best_x, best_f
