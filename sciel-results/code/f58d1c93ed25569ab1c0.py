import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin before 300s kill
        
        # State tracking
        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0
        rng = np.random.default_rng(42)

        def eval_wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return best_f # Return current best to stop optimization gracefully
            x = np.clip(x, -5.0, 5.0)
            f_val = func(x)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Strategy: Multi-Start with Adaptive Local Refinement
        # We divide the budget into "phases".
        # Phase 1: Global Exploration via Random Sampling (high coverage)
        # Phase 2: Local Refinement via Nelder-Mead on the best point found
        
        # Dynamic allocation based on remaining budget
        # We want to ensure we have enough budget for at least one solid NM run.
        # NM in 5D typically needs 50-100 evals to converge well.
        # Let's reserve 150 evals for the final polish, and use the rest for multi-start.
        
        # However, to maximize AOCC, we should interleave or structure it such that
        # we find a decent point quickly, then refine it.
        
        # Approach:
        # 1. Perform a series of random starts.
        # 2. For each start, run a *short* Nelder-Mead to see if it improves.
        # 3. If we find a significantly better point, continue refining.
        # 4. Finally, run a long, high-precision Nelder-Mead on the global best.
        
        # Let's use a "Race" strategy:
        # Split budget into N blocks. In each block, try a random start + short NM.
        # Keep the best. At the end, do a final deep NM.
        
        # For dim=5, let's try 5-10 restarts.
        # Budget 500.
        # If we use 10 restarts, each gets ~40 evals for initial NM.
        # Then final 100 evals for deep NM.
        
        # Actually, a simpler and robust approach for BBOB Sphere:
        # 1. Sample 20 random points.
        # 2. Pick the best 3.
        # 3. Run NM on each of the best 3 with a budget of ~80.
        # 4. Pick the best result.
        # 5. Run a final NM on that best result with remaining budget.
        
        # Let's implement this explicitly.
        
        # Phase 1: Global Scan
        # We have 500 calls.
        # Let's spend 50 calls on random sampling to find good basins.
        num_initial_samples = min(50, budget // 10)
        
        samples = []
        for i in range(num_initial_samples):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.0, 4.0, size=dim) # Min is in [-4,4]^5
            f = eval_wrapper(x)
            samples.append((f, x))
        
        # Sort samples by function value
        samples.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement on top candidates
        # Take top 3 candidates (or fewer if budget is tight)
        top_candidates = samples[:3]
        
        # Allocate remaining budget for NM runs
        # We want to give each candidate a chance to converge.
        # NM in 5D usually converges in ~50-100 iterations.
        # Let's allocate 100 evals per candidate if we have 3.
        # If budget is low, reduce.
        
        remaining_budget = budget - calls
        # Reserve 20% of remaining for final polish?
        # Or just split among candidates?
        
        # Let's try: Split remaining budget among the top 3 candidates equally.
        # But we must ensure we don't exceed budget.
        
        # Actually, a sequential "best-first" approach might be better for AOCC.
        # Refine the best one first. If it converges fast, use leftover for the second best.
        
        # Let's refine the top 3 sequentially.
        # Budget per candidate: max(50, remaining_budget // 3)
        
        for f_init, x_init in top_candidates:
            if calls >= budget or time.time() > deadline:
                break
            
            # Define a local function that stops when budget is low
            # We want to stop NM if it's no longer improving or if we hit time/budget
            # scipy's NM doesn't have a strict "max_calls" that stops immediately on every call,
            # but we can use a wrapper that raises an exception or checks time.
            
            # Better: Use a custom callback to check time/budget?
            # SciPy minimize allows 'maxiter'. 
            # maxiter for NM is roughly (num_vars + 1) * factor.
            # Let's estimate maxiter.
            # If we want to spend ~100 calls, maxiter should be around 100/ (dim+1) * 2?
            # Actually, NM does 1 call per iteration (reflection) + sometimes expansion/contraction.
            # So ~1-3 calls per iteration.
            # Let's set maxiter conservatively.
            
            # We will use a wrapper that tracks calls and stops if we exceed a local budget.
            # To avoid modifying global state in a way that breaks scipy, we'll just let it run
            # but check time in a tight loop? No, scipy is C-optimized mostly.
            
            # Alternative: Implement a simple NM ourselves? No, scipy is reliable.
            # We can use `options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8}`
            
            # How many calls will this take?
            # Let's estimate. If we allow 100 calls, and NM uses ~2 calls/iter, maxiter=50.
            
            # Let's be more dynamic.
            # We have `remaining_budget`.
            # Let's give this candidate `remaining_budget // 3` calls.
            # But we need to convert "calls" to "maxiter".
            # Empirically, NM in 5D uses ~1.5-2.5 function evaluations per iteration.
            # Let's assume 2.0.
            
            local_budget = max(20, (budget - calls) // 3)
            # Cap local budget to not steal from other candidates too much?
            # If we just divide, it's fair.
            
            max_iter = max(10, int(local_budget / 2.0))
            
            # Run Nelder-Mead
            try:
                result = minimize(
                    eval_wrapper, 
                    x_init, 
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
            except Exception:
                pass
            
            # Update best if NM found something better (eval_wrapper already updates global best)
            # But we need to check if we should continue to next candidate.
            # If we found a very good point (e.g., f < 1e-4), we might want to stop exploring others
            # and use remaining budget to polish this one further.
            
            if best_f < 1e-4:
                # We found a very good point. Stop multi-start, go to final polish.
                break

        # Phase 3: Final Polish
        # Use any remaining budget to run a high-precision NM on the global best_x.
        if calls < budget and time.time() < deadline:
            remaining = budget - calls
            # We want to spend as much as possible here.
            # maxiter for NM:
            # If remaining is 200, max_iter ~ 100.
            max_iter_final = max(10, int(remaining / 2.0))
            
            try:
                result = minimize(
                    eval_wrapper, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': max_iter_final, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
            except Exception:
                pass

        return best_x, best_f
