"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, pt):
    """Compute makespan for a given permutation."""
    if len(perm) == 0:
        return 0.0
    # C[i] = max(C[i-1], C_prev_machine[i]) + p[i]
    # We can compute this vectorized or looped. Looping over machines is O(m*n).
    # For small n, this is fine.
    c = 0.0
    for m in range(pt.shape[1]):
        # C on machine m depends on C on machine m-1 and processing times on m
        # Let's just do the standard cumulative sum approach which is O(m*n)
        pass
    
    # Efficient calculation:
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
    # We only care about C[n-1, m-1] (last job, last machine)
    
    # Let's use a simple loop for clarity and correctness first, 
    # but we need speed. 
    # Actually, for SA delta evaluation, we need to be fast.
    # Let's stick to a robust O(m*n) calculation for the base makespan,
    # and optimize the delta if needed. Given 120s limit, even O(n^2 * m) for many steps is ok if n is small.
    
    # Vectorized-ish approach for full makespan:
    # C = np.zeros((n, m))
    # for j in range(n):
    #    for m in range(m):
    #       ...
    
    # Let's just use a loop over jobs, maintaining running completion times per machine.
    # This is O(n*m).
    n = len(perm)
    m = pt.shape[1]
    if n == 0: return 0.0
    
    # C_m is the completion time of the last job processed on machine m
    # Actually, we need the completion time of job j on machine m.
    # Let's maintain an array C where C[m] is the current completion time of the last job on machine m.
    # Wait, that's not right. C[m] needs to be the completion time of the *current* job on machine m.
    # To calculate the next job's completion time on machine m, we need:
    # 1. The completion time of the previous job on machine m (call it prev_C_m)
    # 2. The completion time of the current job on machine m-1 (call it curr_C_m_minus_1)
    # So we need to track the vector of completion times for the current job across all machines?
    # No, we can just iterate.
    
    # Let's use the standard algorithm:
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
    # We can store only the previous row (C[j-1, :]) and the current row being built.
    
    prev_row = np.zeros(m)
    for j in perm:
        curr_row = np.zeros(m)
        for k in range(m):
            if k == 0:
                curr_row[0] = prev_row[0] + pt[j, 0]
            else:
                curr_row[k] = max(prev_row[k], curr_row[k-1]) + pt[j, k]
        prev_row = curr_row
        
    return prev_row[-1]

def _delta_makespan_swap(perm, pt, i, j):
    """
    Calculate the change in makespan if we swap jobs at positions i and j in perm.
    This is complex to do incrementally without recalculating the whole tail.
    Given the complexity of correct delta calculation for flow shop, 
    and the time limit, it might be safer to just recalculate the makespan 
    if the problem size is small, or use a simpler move like insertion.
    
    However, the inspiration suggests O(n) delta. 
    Let's try a simpler approach: just recalculate makespan for the candidate.
    If n is small (e.g. < 100), O(n*m) is very fast.
    If n is large, we need delta.
    
    Let's assume n is moderate. We will use a hybrid: 
    If n < 50, just recalculate.
    Otherwise, we need a better heuristic or delta.
    
    Actually, let's look at the "inspiration 1" which scored 1.12. It used vectorized SA.
    Let's implement a simple SA that recalculates makespan but is very fast in Python/NumPy.
    """
    pass

def order_jobs(processing_times):
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
        
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Initial solution: NEH heuristic (good starting point)
    # 1. Sort jobs by total processing time descending
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # NEH insertion:
    # Start with first job.
    # For each next job, insert it in the position that minimizes makespan.
    # This is O(n^2 * m) for construction, which is acceptable.
    
    current_perm = [order[0]]
    for idx in range(1, n):
        job = order[idx]
        best_pos = 0
        best_mk = float('inf')
        # Try all positions 0 to len(current_perm)
        for pos in range(len(current_perm) + 1):
            # Create new perm
            new_perm = current_perm[:pos] + [job] + current_perm[pos:]
            mk = _makespan(new_perm, processing_times)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    current_mk = best_mk
    best_perm = current_perm[:]
    best_mk = current_mk
    
    # Simulated Annealing
    # Parameters
    T = 100.0 # Initial temperature (scale relative to typical delta? Let's start high)
    # Estimate typical delta: sum of all processing times
    total_p = processing_times.sum()
    T = total_p * 0.1
    cooling = 0.995
    min_T = 1e-6
    
    current_perm = np.array(current_perm)
    n_jobs = len(current_perm)
    
    # Precompute row references for speed?
    # pt is a numpy array.
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Generate neighbor: swap two random positions
        i, j = random.sample(range(n_jobs), 2)
        if i > j: i, j = j, i
        
        # Create candidate
        candidate = current_perm.copy()
        candidate[i], candidate[j] = candidate[j], candidate[i]
        
        # Evaluate candidate
        cand_mk = _makespan(candidate, processing_times)
        delta = cand_mk - current_mk
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_perm = candidate
            current_mk = cand_mk
            
            if current_mk < best_mk:
                best_mk = current_mk
                best_perm = current_perm.copy()
        
        T *= cooling
        
    return list(best_perm)
# EVOLVE-BLOCK-END
