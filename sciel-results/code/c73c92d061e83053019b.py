import numpy as np
from scipy.optimize import differential_evolution
import time

def _calc_makespan(perm, processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    
    # Vectorized approach for speed
    # C[j, m] is completion time of job j on machine m
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
    # This is inherently sequential along the diagonal, so hard to fully vectorize.
    # We stick to loop but optimize inner loop.
    
    C_prev = np.zeros(n_machines)
    for job_idx in perm:
        p = processing_times[job_idx]
        C_curr = np.empty(n_machines)
        for m in range(n_machines):
            if m == 0:
                C_curr[0] = C_prev[0] + p[0]
            else:
                C_curr[m] = max(C_prev[m], C_curr[m-1]) + p[m]
        C_prev = C_curr
    return float(C_prev[-1])

def _decode_weights(weights, n_jobs):
    """
    Decode continuous weights into a permutation.
    Higher weight = earlier position (or vice versa, doesn't matter as long as consistent).
    We use argsort.
    """
    # Ties are broken by index, which is fine.
    return np.argsort(-weights, kind='stable')

def _local_search(perm, processing_times, time_limit=10.0):
    """
    Enhanced Local Search with Insertion and 2-Swap moves.
    Uses Best-Improvement strategy within a time limit.
    """
    n_jobs = len(perm)
    best_perm = list(perm)
    best_makespan = _calc_makespan(best_perm, processing_times)
    start_time = time.time()
    
    # We will alternate between Insertion and 2-Swap to escape local minima slightly better
    # Or just do Insertion as it's generally stronger for flow shop.
    # Let's do a combined neighborhood search.
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        best_new_perm = None
        best_new_makespan = best_makespan
        
        # 1. Try Insertion moves
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            job = best_perm[i]
            for j in range(n_jobs):
                if i == j:
                    continue
                # Create new permutation
                new_perm = best_perm.copy()
                new_perm.pop(i)
                # Adjust insertion index if removing from before
                insert_idx = j - 1 if j > i else j
                new_perm.insert(insert_idx, job)
                
                # Quick check: if the job is already in a similar position, skip?
                # No, just calculate.
                new_makespan = _calc_makespan(new_perm, processing_times)
                
                if new_makespan < best_new_makespan:
                    best_new_makespan = new_makespan
                    best_new_perm = new_perm
        
        # 2. Try 2-Swap moves (if time remains)
        if time.time() - start_time < time_limit - 0.5:
            for i in range(n_jobs):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + 1, n_jobs):
                    new_perm = best_perm.copy()
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                    new_makespan = _calc_makespan(new_perm, processing_times)
                    if new_makespan < best_new_makespan:
                        best_new_makespan = new_makespan
                        best_new_perm = new_perm

        if best_new_perm is not None and best_new_makespan < best_makespan:
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
    
    # 1. Initial Heuristic: Random or Sum-based
    # Let's use a few random starts and a sum-based start
    starts = []
    
    # Heuristic 1: Sum descending
    sums = np.sum(processing_times, axis=1)
    perm_sum = np.argsort(-sums, kind='stable')
    starts.append(perm_sum)
    
    # Heuristic 2: Random
    np.random.seed(42)
    perm_rand = np.random.permutation(n_jobs)
    starts.append(perm_rand)
    
    best_perm = None
    best_makespan = float('inf')
    
    for p in starts:
        ms = _calc_makespan(p, processing_times)
        if ms < best_makespan:
            best_makespan = ms
            best_perm = list(p)
            
    # 2. Optimize Weights using DE
    # We map weights to permutations.
    # We run DE to find weights that yield a good permutation.
    
    def objective(weights):
        perm = _decode_weights(weights, n_jobs)
        return _calc_makespan(perm, processing_times)
        
    bounds = [(0, 1)] * n_jobs
    
    # Adjust DE parameters based on problem size
    # Smaller problems can afford more iterations
    if n_jobs <= 20:
        maxiter = 50
        popsize = 15
    elif n_jobs <= 50:
        maxiter = 30
        popsize = 10
    else:
        maxiter = 20
        popsize = 8
        
    try:
        # Use multiple seeds if time permits? 
        # For now, one run with good settings.
        result = differential_evolution(
            objective, 
            bounds, 
            popsize=popsize, 
            maxiter=maxiter, 
            tol=1e-4,
            seed=42,
            init='sobol',
            polish=False # We do our own LS
        )
        
        best_weights = result.x
        perm_de = _decode_weights(best_weights, n_jobs)
        makespan_de = _calc_makespan(perm_de, processing_times)
        
        if makespan_de < best_makespan:
            best_perm = list(perm_de)
            best_makespan = makespan_de
            
    except Exception:
        pass # Fallback to initial heuristic
    
    # 3. Local Search to polish
    # Allocate remaining time to LS
    # Assume DE took ~5-10s. Total limit 120s.
    # Let's give LS up to 30s to be safe, or calculate remaining.
    # Since we don't track exact DE time easily here without global state, 
    # we'll use a fixed reasonable limit.
    final_perm, _ = _local_search(best_perm, processing_times, time_limit=30.0)
    
    return list(final_perm)
