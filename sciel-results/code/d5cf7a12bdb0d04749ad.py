"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead with adaptive initial step size for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        def time_left():
            return deadline - time.time()
        
        def nm_polish(func, x0, f0, max_calls, deadline, dim):
            """
            Run Nelder-Mead from x0 with f0.
            Returns (best_x, best_f, calls_used).
            """
            if max_calls < 5 or time_left() < 0.5:
                return x0, f0, 0
            
            call_count = [0]
            
            def wrapped_func(x):
                nonlocal call_count
                call_count[0] += 1
                if call_count[0] > max_calls or time_left() < 0.1:
                    raise Exception("Budget exceeded")
                return func(x)
                
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': max_calls, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                calls_used = call_count[0]
                if res.fun < f0:
                    return res.x, res.fun, calls_used
                else:
                    return x0, f0, calls_used
            except Exception:
                return x0, f0, call_count[0]

        # Strategy: Multi-start Nelder-Mead
        # 1. Generate a set of diverse starting points
        # 2. For each, run NM with a limited budget
        # 3. Keep the best result
        
        # Number of starts: we want to use ~40-60 calls per start
        # With budget 500, we can afford ~10 starts
        # But NM can be fast. Let's try 20 starts with 20 calls each?
        # Or 10 starts with 40 calls each?
        # Büche-Rastrigin is multimodal. More starts is better for global coverage.
        # Let's try 25 starts, budget 15 each. Total 375 calls.
        # Or 50 starts, budget 8 each. Total 400 calls.
        
        # Let's use a stratified sampling approach for initial points
        # Divide the box into a grid
        # dim=5, 25 points -> 2^5=32 grid points is close
        # Let's use 32 grid points (corners of a 2x2x2x2x2 grid)
        # Plus the center
        # Plus some random points
        
        starts = []
        
        # Grid points
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        for m in range(2):
                            x = np.array([
                                -5 + 10*i,
                                -5 + 10*j,
                                -5 + 10*k,
                                -5 + 10*l,
                                -5 + 10*m
                            ])
                            starts.append(x)
        
        # Center
        starts.append(np.zeros(dim))
        
        # Random points
        for _ in range(10):
            x = rng.uniform(-5, 5, size=dim)
            starts.append(x)
        
        # Shuffle starts to randomize the order of evaluation
        # But we want to evaluate the most promising ones first?
        # Actually, since we don't know which is best, random order is fine.
        # Or we can evaluate all starts first (1 call each), then run NM on the best few?
        # That's a better strategy:
        # 1. Evaluate all starts (33 calls)
        # 2. Sort by f value
        # 3. Run NM on the top K starts with remaining budget
        
        # Let's do that.
        
        # Evaluate all starts
        f_starts = np.array([func(x) for x in starts])
        calls_made += len(starts)
        
        # Update best
        idx_min = np.argmin(f_starts)
        best_f = f_starts[idx_min]
        best_x = starts[idx_min].copy()
        
        # Sort starts by f value
        sorted_idx = np.argsort(f_starts)
        
        # Run NM on the top K starts
        # Remaining budget: budget - calls_made
        # Let's try to run NM on the top 10 starts
        # Budget per start: (budget - calls_made) / 10
        # But we should give more budget to the best starts
        
        # Let's try a weighted allocation
        # Top 1 gets 50% of remaining, next 4 get 10% each, next 5 get 5% each
        # This is complex. Let's just try a fixed number of starts with a fixed budget.
        
        # Let's try 5 starts with 80 calls each.
        # 5 * 80 = 400 calls. Plus 33 initial = 433 calls.
        # Leaves 67 calls for a final polish on the best result.
        
        k_starts = 5
        nm_budget_each = 80
        
        for i in range(k_starts):
            if calls_made + nm_budget_each > budget:
                break
            if time_left() < 1.0:
                break
                
            idx = sorted_idx[i]
            x_start = starts[idx]
            f_start = f_starts[idx]
            
            x_nm, f_nm, used_nm = nm_polish(func, x_start, f_start, nm_budget_each, deadline, dim)
            calls_made += used_nm
            
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm.copy()
        
        # Final polish on the best result
        remaining = budget - calls_made
        if remaining > 10 and time_left() > 5.0:
            x_final, f_final, used_final = nm_polish(func, best_x, best_f, remaining, deadline, dim)
            calls_made += used_final
            
            if f_final < best_f:
                best_f = f_final
                best_x = x_final.copy()
        
        return best_x, best_f

# EVOLVE-BLOCK-END
