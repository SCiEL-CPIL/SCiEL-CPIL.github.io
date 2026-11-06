"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, processing_times):
    """Compute makespan for a given job order."""
    n_machines = processing_times.shape[1]
    if n_machines == 0 or len(order) == 0:
        return 0.0
    completion = np.zeros(n_machines)
    for job in order:
        p = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            prev = max(prev, completion[m]) + p[m]
            completion[m] = prev
    return float(completion[-1])


def _neh_initialization(processing_times):
    """
    NEH heuristic: insert jobs one by one in order of decreasing total processing time,
    choosing the position that minimizes the makespan at each step.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    # Order jobs by decreasing total processing time
    totals = processing_times.sum(axis=1)
    job_order = list(np.argsort(-totals))
    
    # Build sequence greedily
    seq = [job_order[0]]
    current_makespan = _makespan(seq, processing_times)
    
    for job in job_order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
        current_makespan = best_ms
    
    return seq


def _improve_adjacent_swaps(order, processing_times, time_limit=110.0):
    """
    Greedy local search using only adjacent swaps.
    This is significantly faster than 2-opt/Or-opt.
    """
    start_time = time.time()
    n_jobs = len(order)
    if n_jobs <= 1:
        return order
    
    best_order = list(order)
    best_ms = _makespan(best_order, processing_times)
    
    while time.time() - start_time < time_limit:
        improved = False
        # Check all adjacent swaps
        for i in range(n_jobs - 1):
            # Swap i and i+1
            if time.time() - start_time >= time_limit:
                break
            new_order = best_order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            ms = _makespan(new_order, processing_times)
            if ms < best_ms - 1e-9:
                best_order = new_order
                best_ms = ms
                improved = True
        if not improved:
            break
    
    return best_order


def order_jobs(processing_times):
    """
    Main function: construct a job permutation minimizing makespan.
    Uses NEH initialization, block-based decomposition, and local search.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: NEH initialization
    initial_order = _neh_initialization(processing_times)
    best_order = list(initial_order)
    best_ms = _makespan(best_order, processing_times)
    
    # Step 2: Local search polish using adjacent swaps
    if time.time() < deadline - 2.0:
        remaining_time = deadline - time.time() - 2.0
        best_order = _improve_adjacent_swaps(best_order, processing_times, time_limit=remaining_time)
    
    return best_order
# EVOLVE-BLOCK-END
