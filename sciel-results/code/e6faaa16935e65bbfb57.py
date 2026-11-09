import numpy as np
from scipy.optimize import differential_evolution
import time

def _calc_makespan(perm, processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    C_prev = np.zeros(n_machines)
    for job_idx in perm:
        C_curr = np.zeros(n_machines)
        # Vectorized calculation for a single job across machines
        # C_curr[m] = max(C_prev[m], C_curr[m-1]) + p[job_idx, m]
        # This is inherently sequential due to C_curr[m-1] dependency, 
        # but we can write it efficiently in a loop.
        for m in range(n_machines):
            start = C_prev[m]
            if m > 0:
                start = max(start, C_curr[m-1])
            C_curr[m] = start + processing_times[job_idx, m]
        C_prev = C_curr
    return float(C_prev[-1])

def _perm_from_weights(weights):
    # Stable sort by weights descending
    return np.argsort(-weights, kind='stable')

def _local_search_insertion(perm, processing_times, time_limit):
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm, _calc_makespan(perm, processing_times)
    
    best_perm = list(perm)
    best_makespan = _calc_makespan(best_perm, processing_times)
    start_time = time.time()
    
    # Use a tabu-like or simple best-improvement insertion
    # To be faster, we can precompute some things, but for small N, simple is fine.
    # We will use "Best Improvement" strategy within the time limit.
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        best_new_perm = None
        best_new_makespan = best_makespan
        
        # Iterate over all removal and insertion positions
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            job = best_perm[i]
            # Remove job
            remaining = [x for k, x in enumerate(best_perm) if k != i]
            
            for j in range(len(remaining) + 1):
                # Insert job at j
                new_perm = remaining[:j] + [job] + remaining[j:]
                
                # Quick check: if new_perm is same as best_perm (can happen if i==j effectively)
                # Actually, removing and inserting at same relative spot yields same perm.
                # We can skip if i == j or i == j+1 (depending on index logic)
                # But it's easier to just check makespan.
                
                # Optimization: Skip if this is the original permutation
                # We can check if new_perm == best_perm, but that's O(N).
                # Let's just calculate makespan.
                
                ms = _calc_makespan(new_perm, processing_times)
                
                if ms < best_new_makespan:
                    best_new_makespan = ms
                    best_new_perm = new_perm
        
        if best_new_perm is not None:
            best_perm = best_new_perm
            best_makespan = best_new_makespan
            improved = True
        else:
            break
            
    return best_perm, best_makespan

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    start_time = time.time()
    total_time_limit = 110.0 # Leave 10s buffer
    
    # 1. Initial Heuristic: 
    # Instead of simple sum, use a critical path heuristic or random restarts.
    # Let's generate a few random initial weights and pick the best.
    best_perm = None
    best_makespan = float('inf')
    
    # Try a few constructive heuristics
    heuristics = [
        lambda: np.argsort(-np.sum(processing_times, axis=1), kind='stable'),
        lambda: np.argsort(np.sum(processing_times, axis=1), kind='stable'),
        lambda: np.argsort(processing_times[:, -1], kind='stable'), # Last machine priority
        lambda: np.argsort(processing_times[:, 0], kind='stable'),  # First machine priority
        lambda: np.random.rand(n_jobs) # Random
    ]
    
    for h in heuristics:
        try:
            w = h()
            perm = _perm_from_weights(w)
            ms = _calc_makespan(perm, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_perm = perm
        except Exception:
            pass
            
    # 2. Global Optimization using Differential Evolution on Weights
    # Allocate ~70% of time to DE, ~30% to Local Search
    de_time_limit = total_time_limit * 0.7
    ls_time_limit = total_time_limit * 0.3
    
    if n_jobs <= 50: # DE is feasible for small/medium N
        def objective(weights):
            perm = _perm_from_weights(weights)
            return _calc_makespan(perm, processing_times)
            
        bounds = [(0, 1)] * n_jobs
        
        # Use a callback to stop early if time runs out? 
        # SciPy DE doesn't have a direct time callback, but we can estimate iterations.
        # Or we can just run it and hope it finishes. 
        # Better: Use a wrapper that raises an exception if time limit exceeded?
        # That's risky.
        # Alternative: Run DE with fixed maxiter, but choose maxiter based on N and time.
        # For N=20, one eval is fast. For N=50, slower.
        
        # Let's try to run DE.
        try:
            # Popsize 15, Maxiter 50 is a reasonable start.
            # If N is small, we can do more.
            maxiter = 100 if n_jobs < 20 else 50
            popsize = 15
            
            result = differential_evolution(
                objective, 
                bounds, 
                popsize=popsize, 
                maxiter=maxiter,
                tol=1e-5,
                seed=42,
                init='sobol',
                polish=False # We do our own polish
            )
            
            if result.success or result.fun < best_makespan:
                best_weights = result.x
                best_perm = _perm_from_weights(best_weights)
                best_makespan = _calc_makespan(best_perm, processing_times)
                
        except Exception:
            pass # Keep best from heuristics
            
    # 3. Local Search Polish
    current_time = time.time()
    remaining_time = total_time_limit - (current_time - start_time)
    if remaining_time > 1:
        best_perm, best_makespan = _local_search_insertion(
            list(best_perm), 
            processing_times, 
            time_limit=remaining_time
        )
        
    return list(best_perm)
