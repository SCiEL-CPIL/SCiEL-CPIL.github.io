import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Keep 30s margin for safety against the 300s/330s limit
        deadline = start_time + 290
        
        n = self.dim
        total_budget = self.budget
        
        # State variables
        calls_made = 0
        best_x = np.full(n, np.nan)
        best_f = np.inf
        
        # Helper to safely call the function
        def do_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def time_ok():
            return time.time() < deadline

        # --- Phase 1: Global Sampling / Multi-Start Initialization ---
        # We use a portion of the budget to scatter points across the domain.
        # This acts as a coarse global search to identify potential basins.
        # For Rastrigin, uniform sampling is often sufficient to find several local minima.
        # We allocate ~20% of the budget to this phase, ensuring we have enough
        # for the subsequent NM polishing.
        
        # We want to start 5 independent NM optimizers.
        # Let's allocate budget for 5 NM runs.
        # Each NM run on dim=5 typically needs ~50-100 evaluations to converge well.
        # Total budget is 500. 
        # If we use 50 for sampling, we have 450 left.
        # 450 / 5 = 90 calls per NM run. This is reasonable.
        
        num_starts = 5
        sampling_budget = int(total_budget * 0.15) # ~75 calls
        if sampling_budget < num_starts * n: # Ensure we can at least evaluate the starts
             sampling_budget = num_starts * n + 10

        rng = np.random.default_rng(42)
        
        # Generate initial candidates for the multi-start
        # We mix uniform randoms with some structured points (corners, center) 
        # to ensure coverage.
        candidates = []
        
        # 1. Center
        candidates.append(np.zeros(n))
        
        # 2. Corners of the box [-5, 5]
        # We can't do all 2^n corners if n is large, but n=5 is small (32 corners).
        # However, Rastrigin is periodic-ish. Let's just use random points mostly.
        # But adding a few deterministic points helps reproducibility/coverage.
        
        # Add random points
        num_random = max(0, sampling_budget - 1)
        random_points = rng.uniform(-5, 5, (num_random, n))
        candidates.extend(random_points.tolist())
        
        # Evaluate all candidates
        candidate_points = []
        candidate_fvals = []
        
        for x in candidates:
            if not time_ok() or calls_made >= sampling_budget:
                break
            f_val = do_call(x)
            if np.isfinite(f_val):
                candidate_points.append(x)
                candidate_fvals.append(f_val)
                
        if not candidate_points:
            # Fallback if something went wrong
            return best_x, best_f

        # Sort by fitness to pick the best distinct starting points
        # We want to start NM from the best few, but ensure they are not too close
        # to each other to cover different basins.
        
        indices = np.argsort(candidate_fvals)
        
        selected_starts = []
        min_dist = 1.0 # Minimum distance between start points
        
        for idx in indices:
            if len(selected_starts) >= num_starts:
                break
            x_candidate = candidate_points[idx]
            # Check distance against already selected starts
            is_distinct = True
            for x_sel in selected_starts:
                if np.linalg.norm(x_candidate - x_sel) < min_dist:
                    is_distinct = False
                    break
            if is_distinct:
                selected_starts.append(x_candidate)
                
        # If we didn't find enough distinct starts, just take the top ones regardless
        while len(selected_starts) < num_starts and len(candidate_points) > len(selected_starts):
             # Add next best even if close
             idx = indices[len(selected_starts)]
             if candidate_points[idx] not in selected_starts:
                 selected_starts.append(candidate_points[idx])

        # --- Phase 2: Parallel (Sequential) Nelder-Mead Polishing ---
        # We run NM from each selected start point.
        # We split the remaining budget among the starts.
        
        remaining_budget = total_budget - calls_made
        if remaining_budget <= 0 or not time_ok():
            return best_x, best_f
            
        budget_per_start = max(10, remaining_budget // num_starts)
        
        # We will iterate through the starts. 
        # To allow "parallel" behavior in a single thread, we just run them sequentially.
        # However, if one NM run finishes early (converges), we can use the leftover 
        # budget for the others or re-start from the new best.
        
        # To keep it simple and robust:
        # 1. Run NM for each start with its allocated budget.
        # 2. Track the best result.
        # 3. If time/budget remains, run a final NM from the absolute best found so far.
        
        nm_results = []
        
        for i, x0 in enumerate(selected_starts):
            if not time_ok() or calls_made >= total_budget:
                break
                
            # Wrap func for this specific NM run to enforce local budget
            local_calls = [0]
            local_max_calls = budget_per_start
            
            def nm_func_local(x_nm):
                nonlocal local_calls
                if local_calls[0] >= local_max_calls:
                    # Return a large value to signal stop, or just the current best?
                    # Returning the current best might confuse NM if it's not improving.
                    # Better to just stop the optimization if we hit the limit.
                    # Scipy's minimize will stop if we raise an exception or if we 
                    # modify the result. 
                    # Actually, simplest is to let it run but ensure we don't exceed global budget.
                    pass
                # Check global constraints
                if calls_made >= total_budget or time.time() > deadline:
                    return best_f # Return best known to halt progress
                val = do_call(x_nm)
                local_calls[0] += 1
                return val
            
            try:
                # Nelder-Mead options
                # xatol and fatol help stop early if converged
                res = minimize(nm_func_local, x0, method='Nelder-Mead', 
                               options={'maxiter': local_max_calls, 
                                        'xatol': 1e-6, 
                                        'fatol': 1e-6,
                                        'adaptive': True})
                
                # Update global best if this run found something better
                # Note: do_call already updates best_f and best_x internally
                # So we just need to ensure we didn't crash.
                
            except Exception as e:
                # If NM fails, just continue to next start
                pass
            
            # Check if we should stop early if we found a very good solution
            # e.g., if best_f is very close to 0 (Rastrigin min is 0)
            # But f_opt is unknown and shifted. So we can't use absolute threshold.
            # We just rely on the budget.

        # --- Phase 3: Final Polish ---
        # If we have budget left, run one more NM from the global best_x
        if calls_made < total_budget and time_ok() and np.isfinite(best_f):
            final_budget = total_budget - calls_made
            if final_budget > 10:
                final_calls = [0]
                
                def nm_func_final(x_nm):
                    nonlocal final_calls
                    if final_calls[0] >= final_budget:
                        return best_f
                    if calls_made >= total_budget or time.time() > deadline:
                        return best_f
                    val = do_call(x_nm)
                    final_calls[0] += 1
                    return val

                try:
                    res = minimize(nm_func_final, best_x, method='Nelder-Mead', 
                                   options={'maxiter': final_budget, 
                                            'xatol': 1e-8, 
                                            'fatol': 1e-8,
                                            'adaptive': True})
                except Exception:
                    pass

        return best_x, best_f
