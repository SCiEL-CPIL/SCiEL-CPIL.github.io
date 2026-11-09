"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    # Use vectorized approach for speed
    # completion[i][j] = completion of job i on machine j
    # For job i in order, machine j:
    #   if j == 0: C[i][0] = C[i-1][0] + p[i][0]
    #   else: C[i][j] = max(C[i-1][j], C[i][j-1]) + p[i][j]
    
    # We'll compute this iteratively but in a somewhat vectorized way
    # Actually for flow shop, we need sequential computation along both dimensions
    
    # Let's use the standard O(n*m) algorithm
    c = np.zeros((n_jobs, n_machines))
    for i in range(n_jobs):
        job_idx = order[i]
        for j in range(n_machines):
            p = processing_times[job_idx, j]
            if i == 0 and j == 0:
                c[i, j] = p
            elif i == 0:
                c[i, j] = c[i, j-1] + p
            elif j == 0:
                c[i, j] = c[i-1, j] + p
            else:
                c[i, j] = max(c[i-1, j], c[i, j-1]) + p
    
    return c[-1, -1]

def _neh_rule(processing_times):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # Insert each job into the best position
    order = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            candidate = order[:pos] + [job] + order[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order.insert(best_pos, job)
    
    return order

def _local_search(processing_times, order, max_iter=10000, time_limit=10.0):
    """2-opt local search with time limit."""
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan(processing_times, order)
    
    best_order = list(order)
    best_ms = _makespan(processing_times, best_order)
    start_time = time.time()
    
    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        for i in range(n_jobs - 1):
            for j in range(i + 1, n_jobs):
                # Try swapping
                new_order = list(best_order)
                new_order[i], new_order[j] = new_order[j], new_order[i]
                ms = _makespan(processing_times, new_order)
                if ms < best_ms:
                    best_order = new_order
                    best_ms = ms
                    improved = True
        
        # Also try 2-opt (reverse segment)
        for i in range(n_jobs - 1):
            for j in range(i + 1, n_jobs):
                new_order = list(best_order)
                new_order[i:j+1] = new_order[i:j+1][::-1]
                ms = _makespan(processing_times, new_order)
                if ms < best_ms:
                    best_order = new_order
                    best_ms = ms
                    improved = True
    
    return best_order, best_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    
    Uses NEH heuristic followed by local search with time management.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin from 150s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Generate initial solution using NEH
    initial_order = _neh_rule(processing_times)
    initial_ms = _makespan(processing_times, initial_order)
    
    # Run local search with time budget
    remaining_time = deadline - time.time()
    if remaining_time < 5.0:
        return initial_order
    
    time_limit = min(remaining_time - 5.0, 60.0)
    best_order, best_ms = _local_search(
        processing_times, initial_order, 
        max_iter=50000, 
        time_limit=time_limit
    )
    
    # Check if we have time for more iterations or restarts
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        # Try multiple random restarts with local search
        best_overall = best_order
        best_overall_ms = best_ms
        
        num_restarts = min(int(remaining_time / 5.0), 10)
        for _ in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Random initial order
            random_order = list(range(n_jobs))
            random.shuffle(random_order)
            
            # Quick local search with shorter time limit
            tl = min(remaining_time - 5.0, 10.0)
            if tl <= 0:
                break
            order, ms = _local_search(
                processing_times, random_order,
                max_iter=10000,
                time_limit=tl
            )
            
            if ms < best_overall_ms:
                best_overall = order
                best_overall_ms = ms
            
            remaining_time = deadline - time.time()
            if remaining_time <= 5.0:
                break
    
    return best_overall
# EVOLVE-BLOCK-END
