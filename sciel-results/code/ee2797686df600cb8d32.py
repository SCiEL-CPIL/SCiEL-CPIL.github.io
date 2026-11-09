"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, pt):
    """Compute makespan for a given permutation."""
    if len(perm) == 0:
        return 0.0
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
    # We only need the last job's completion time on the last machine.
    # Let's use an array to store completion times for the current machine.
    n = len(perm)
    m = pt.shape[1]
    # c[m] is the completion time of the previous job on machine m
    # Actually, we need C[j, m-1] which is the completion time of job j on machine m-1.
    # And C[j-1, m] which is the completion time of job j-1 on machine m.
    
    # Let's maintain an array `c` where c[m] is the completion time of the last processed job on machine m.
    # But we also need the completion time of the current job on the previous machine.
    # Let's use a different approach:
    # For each job j in perm:
    #   prev_c = 0
    #   for each machine k:
    #      c[k] = max(c[k], prev_c) + pt[perm[j], k]
    #      prev_c = c[k]
    #   The makespan is c[m-1] after the last job.
    
    c = np.zeros(m)
    for job_idx in perm:
        prev_c = 0.0
        for k in range(m):
            c[k] = max(c[k], prev_c) + pt[job_idx, k]
            prev_c = c[k]
    return c[-1]

def _delta_makespan_swap(perm, i, j, pt):
    """
    Calculate the change in makespan if we swap positions i and j in perm.
    This is complex to compute exactly without recalculating, so we might just recalculate
    for simplicity first, or use a heuristic.
    Given the small size of n (usually < 100), recalculation is O(n*m) which is fast.
    However, for SA we need many evaluations.
    Let's stick to recalculation for correctness and simplicity, and optimize if needed.
    """
    new_perm = perm.copy()
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return _makespan(new_perm, pt) - _makespan(perm, pt)

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin for 150s limit
    
    n, m = processing_times.shape
    
    if n <= 1:
        return list(range(n))
        
    # 1. Initial Solution: NEH (Nawaz-Enscore-Ham)
    # Sort jobs by total processing time in descending order.
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = []
    for job in order:
        # Find the best position to insert the job
        best_pos = -1
        best_ms = float('inf')
        for k in range(len(perm) + 1):
            trial_perm = perm[:k] + [job] + perm[k:]
            ms = _makespan(trial_perm, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = k
        perm.insert(best_pos, job)
        
    best_perm = perm.copy()
    best_ms = best_ms
    
    # 2. Simulated Annealing
    current_perm = best_perm.copy()
    current_ms = best_ms
    
    # Initial temperature
    T = best_ms * 0.1
    T_min = best_ms * 0.001
    alpha = 0.9995  # Cooling rate
    
    # Precompute random neighbors? No, just random swaps.
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate a neighbor by swapping two random positions
        i, j = random.sample(range(n), 2)
        new_perm = current_perm.copy()
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        
        new_ms = _makespan(new_perm, processing_times)
        delta = new_ms - current_ms
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_perm = new_perm
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm.copy()
                
        T *= alpha
        
    return list(best_perm)
# EVOLVE-BLOCK-END
