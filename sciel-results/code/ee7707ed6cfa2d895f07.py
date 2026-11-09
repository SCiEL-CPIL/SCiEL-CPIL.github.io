"""Flow-shop makespan minimizer: Optimized NEH + Fast Local Search."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _calc_makespan(perm, pt):
    """
    Optimized makespan calculation.
    Uses a single loop over jobs, maintaining the completion time of the previous job on each machine.
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    
    # C[m] is the completion time of the previous job on machine m
    # Initially 0 for all machines
    C = np.zeros(n_machines, dtype=np.float64)
    
    for j in range(n_jobs):
        job_idx = perm[j]
        p = pt[job_idx]
        # For each machine, update completion time
        # C_new[m] = max(C_old[m], C_new[m-1]) + p[m]
        # We can do this in-place if we are careful, but it's easier to just use a loop
        # Since n_machines is usually small (10-100), this is fast.
        for m in range(n_machines):
            if m == 0:
                C[m] = C[m] + p[m]
            else:
                C[m] = max(C[m], C[m-1]) + p[m]
                
    return float(C[-1])

def _neh_heuristic(processing_times):
    """
    NEH heuristic: Sort by total processing time descending, insert optimally.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return np.array([], dtype=np.int64)
    
    # Sort jobs by total processing time descending
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals).astype(np.int64)
    
    perm = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_cost = float('inf')
        
        # Try inserting at each position
        for pos in range(i + 1):
            # Create candidate permutation
            # To avoid copying the whole list every time, we can be smarter,
            # but for n_jobs < 1000, list operations are fine.
            cand = perm[:pos] + [job] + perm[pos:]
            cost = _calc_makespan(np.array(cand, dtype=np.int64), processing_times)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        
        perm = perm[:best_pos] + [job] + perm[best_pos:]
        
    return np.array(perm, dtype=np.int64)

def _local_search_insertion(perm, pt, deadline):
    """
    Local search using insertion moves.
    Iterates until no improvement or time limit.
    """
    n_jobs = len(perm)
    best = perm.copy()
    best_cost = _calc_makespan(best, pt)
    
    improved = True
    while improved:
        improved = False
        if time.monotonic() > deadline:
            break
            
        for i in range(n_jobs):
            if time.monotonic() > deadline:
                break
            job = best[i]
            # Remove job
            rest = np.delete(best, i)
            
            # Try inserting back at each position
            for j in range(n_jobs):
                if j == i:
                    continue
                # Insert at j in 'rest'
                # If j < i, inserting at j in rest is same as j in original
                # If j > i, inserting at j-1 in rest is same as j in original
                # Actually, simpler: just construct the new array
                cand = np.concatenate([rest[:j], [job], rest[j:]])
                cost = _calc_makespan(cand, pt)
                
                if cost < best_cost - 1e-12:
                    best = cand
                    best_cost = cost
                    improved = True
                    break # Restart scan after improvement
        
    return best, best_cost

def order_jobs(processing_times):
    processing_times = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    t0 = time.monotonic()
    # Strict deadline: 110 seconds to ensure safety margin under 120s limit
    deadline = t0 + 110.0
    
    # 1. Generate initial solution using NEH
    perm = _neh_heuristic(processing_times)
    cost = _calc_makespan(perm, processing_times)
    
    # 2. Local Search
    if time.monotonic() < deadline - 5.0:
        perm, cost = _local_search_insertion(perm, processing_times, deadline - 2.0)
        
    return list(perm)
# EVOLVE-BLOCK-END
