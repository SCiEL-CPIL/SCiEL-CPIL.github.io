"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    # Use vectorized DP for makespan
    # completion[j][m] = max(completion[j-1][m], completion[j][m-1]) + pt[perm[j]][m]
    # We can compute row by row
    prev_row = np.zeros(n_machines)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            # cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
            # This is a sequential recurrence, can't fully vectorize easily
            # But for typical small n_machines it's fine
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
    return float(prev_row[-1])


def _neh_order(processing_times):
    """Generate NEH heuristic ordering."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Forward NEH: insert jobs one by one at best position
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(partial) + 1):
            candidate = partial[:pos] + [job] + partial[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        partial = partial[:best_pos] + [job] + partial[best_pos:]
    
    return partial


def _neighborhood_search(perm, processing_times, max_iterations=500):
    """Simple 2-opt style neighborhood search (swap adjacent jobs)."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    
    for _ in range(max_iterations):
        improved = False
        for i in range(n_jobs - 1):
            # Try swapping adjacent jobs i and i+1
            new_perm = list(best_perm)
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            ms = _makespan(new_perm, processing_times)
            if ms < best_ms - 1e-10:
                best_perm = new_perm
                best_ms = ms
                improved = True
        if not improved:
            break
    
    return best_perm


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin: 10s before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    # Baseline: longest processing time first
    baseline_perm = list(np.argsort(-processing_times.sum(axis=1)))
    best_perm = baseline_perm
    best_ms = _makespan(best_perm, processing_times)
    
    # Try NEH heuristic
    if time.time() < deadline:
        neh_perm = _neh_order(processing_times)
        neh_ms = _makespan(neh_perm, processing_times)
        if neh_ms < best_ms:
            best_perm = neh_perm
            best_ms = neh_ms
    
    # Local search polish on best solution
    if time.time() < deadline and n_jobs > 1:
        # Use remaining time to do as many local search iterations as possible
        elapsed = time.time() - start_time
        # Estimate time per iteration: makespan computation is O(n_jobs * n_machines)
        # For each swap, we recompute makespan. Be conservative.
        if n_jobs <= 50:
            max_iter = 1000
        elif n_jobs <= 100:
            max_iter = 200
        else:
            max_iter = 50
        
        polished = _neighborhood_search(best_perm, processing_times, max_iterations=max_iter)
        polished_ms = _makespan(polished, processing_times)
        if polished_ms < best_ms:
            best_perm = polished
            best_ms = polished_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
