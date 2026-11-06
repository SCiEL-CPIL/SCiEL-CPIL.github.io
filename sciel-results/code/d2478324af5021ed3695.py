"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _calc_makespan(processing_times, perm):
    """Calculate makespan for a given permutation."""
    n_machines = processing_times.shape[1]
    completion = np.zeros(n_machines)
    for j in perm:
        for m in range(n_machines):
            completion[m] = max(completion[m], completion[m-1] if m > 0 else 0) + processing_times[j, m]
    return completion[-1]


def _calc_makespan_array(processing_times, perm_array):
    """Calculate makespan using numpy arrays for speed."""
    n_machines = processing_times.shape[1]
    completion = np.zeros(n_machines)
    for j in perm_array:
        for m in range(n_machines):
            completion[m] = max(completion[m], completion[m-1] if m > 0 else 0) + processing_times[j, m]
    return completion[-1]


def _swap_makespan(processing_times, perm, i, j, current_ms):
    """Calculate makespan after swapping positions i and j, with early termination."""
    n_machines = processing_times.shape[1]
    # Build completion matrix for prefix up to min(i,j)
    a, b = min(i, j), max(i, j)
    
    # Compute completions up to position a
    comp = np.zeros(n_machines)
    for k in range(a):
        job = perm[k]
        for m in range(n_machines):
            comp[m] = max(comp[m], comp[m-1] if m > 0 else 0) + processing_times[job, m]
    
    # Now process positions a through b-1 with the swap
    # Position a gets job that was at b, positions a+1..b-1 stay, position b gets job that was at a
    job_a = perm[a]
    job_b = perm[b]
    
    for k in range(a, b + 1):
        if k == a:
            job = job_b
        elif k == b:
            job = job_a
        else:
            job = perm[k]
        
        for m in range(n_machines):
            comp[m] = max(comp[m], comp[m-1] if m > 0 else 0) + processing_times[job, m]
    
    # Process remaining positions
    for k in range(b + 1, len(perm)):
        job = perm[k]
        for m in range(n_machines):
            comp[m] = max(comp[m], comp[m-1] if m > 0 else 0) + processing_times[job, m]
    
    return comp[-1]


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    # NEH heuristic: order jobs by total processing time descending
    # For each job in order, insert it at the best position in the current sequence
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _calc_makespan(processing_times, new_perm)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        perm.insert(best_pos, job)
    
    best_perm = perm[:]
    best_ms = _calc_makespan(processing_times, best_perm)
    
    # Local search: 2-opt style swaps with incremental evaluation
    n_jobs_int = int(n_jobs)
    while time.time() < deadline:
        improved = False
        for i in range(n_jobs_int - 1):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n_jobs_int):
                # Use incremental makespan calculation
                ms = _swap_makespan(processing_times, best_perm, i, j, best_ms)
                if ms < best_ms:
                    best_ms = ms
                    best_perm[i], best_perm[j] = best_perm[j], best_perm[i]
                    improved = True
        if not improved:
            break
    
    # Final polish: try adjacent swaps with tighter check
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(n_jobs_int - 1):
                if time.time() >= deadline - 2:
                    break
                j = i + 1
                ms = _swap_makespan(processing_times, best_perm, i, j, best_ms)
                if ms < best_ms:
                    best_ms = ms
                    best_perm[i], best_perm[j] = best_perm[j], best_perm[i]
                    improved = True
    
    return list(best_perm)
# EVOLVE-BLOCK-END
