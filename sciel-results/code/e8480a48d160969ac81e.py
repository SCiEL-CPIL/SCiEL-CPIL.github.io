"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(processing_times, order):
    """Compute makespan for a given job order using the flow-shop recurrence."""
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    # c[j] = completion time of current job on machine j
    c = np.zeros(n_machines)
    for job_idx in order:
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            if j == 0:
                c[j] = c[j] + p_row[j]
            else:
                c[j] = max(c[j], c[j-1]) + p_row[j]
    return c[-1]


def _neh_rule(processing_times):
    """NEH heuristic: insert jobs one by one at the best position."""
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    # Sort by total processing time descending
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    # Build sequence incrementally
    seq = [int(sorted_jobs[0])]
    
    for i in range(1, n_jobs):
        job = int(sorted_jobs[i])
        best_pos = 0
        best_span = float('inf')
        
        # Try inserting at each position
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [job] + seq[pos:]
            span = _makespan(processing_times, candidate)
            if span < best_span:
                best_span = span
                best_pos = pos
        
        seq.insert(best_pos, job)
    
    return seq


def _local_search(processing_times, order, time_limit):
    """First-improvement insertion local search with time limit."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs <= 1:
        return order
    
    best_order = list(order)
    best_span = _makespan(processing_times, best_order)
    start_time = time.time()
    deadline = start_time + time_limit
    
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        # Try moving each job to a different position
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            job = best_order[i]
            # Try inserting at each other position
            for j in range(n_jobs):
                if j == i or j == i + 1:
                    continue
                # Remove job from position i
                if j > i:
                    new_order = best_order[:i] + best_order[i+1:j+1] + [job] + best_order[j+1:]
                else:
                    new_order = best_order[:j] + [job] + best_order[j:i] + best_order[i+1:]
                
                span = _makespan(processing_times, new_order)
                if span < best_span:
                    best_span = span
                    best_order = new_order
                    improved = True
                    # Restart the outer loop since order changed
                    i = n_jobs  # break inner
                    break
            if improved:
                break
    
    return best_order


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan using NEH + local search."""
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    total_time_limit = 115.0  # Safety margin before 120s limit
    
    # Step 1: NEH heuristic to get initial solution
    neh_order = _neh_rule(processing_times)
    neh_span = _makespan(processing_times, neh_order)
    
    # Also try simple LPT as fallback/alternative
    lpt_order = list(np.argsort(-processing_times.sum(axis=1)))
    lpt_span = _makespan(processing_times, lpt_order)
    
    # Choose better of NEH and LPT
    if lpt_span < neh_span:
        best_order = lpt_order
        best_span = lpt_span
    else:
        best_order = neh_order
        best_span = neh_span
    
    # Step 2: Local search with remaining time
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    if remaining > 1.0:
        # Use most of the remaining time for local search
        ls_time = remaining * 0.9
        improved_order = _local_search(processing_times, best_order, ls_time)
        improved_span = _makespan(processing_times, improved_order)
        
        if improved_span < best_span:
            best_order = improved_order
            best_span = improved_span
    
    return best_order
# EVOLVE-BLOCK-END
