"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog


def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    # C[i][j] = completion time of job i on machine j
    # We can compute this in O(n_jobs * n_machines)
    completion = np.zeros((n_jobs, n_machines))
    for i, job_idx in enumerate(order):
        for j in range(n_machines):
            p = processing_times[job_idx, j]
            if j == 0:
                if i == 0:
                    completion[i, j] = p
                else:
                    completion[i, j] = completion[i-1, j] + p
            else:
                if i == 0:
                    completion[i, j] = completion[i, j-1] + p
                else:
                    completion[i, j] = max(completion[i-1, j], completion[i, j-1]) + p
    
    return completion[-1, -1]


def _neh_rule(processing_times):
    """NEH heuristic for flow-shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    
    # Sort jobs by total processing time in descending order
    total_pt = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_pt)
    
    # Start with the first two jobs, choose the better order
    order = []
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [sorted_jobs[0]]
    
    # Choose initial pair
    j1, j2 = sorted_jobs[0], sorted_jobs[1]
    # Compute makespan for both orders of the pair
    ms_12 = _makespan(processing_times, [j1, j2])
    ms_21 = _makespan(processing_times, [j2, j1])
    
    if ms_12 <= ms_21:
        order = [j1, j2]
    else:
        order = [j2, j1]
    
    # Insert remaining jobs one by one
    for k in range(2, n_jobs):
        job = sorted_jobs[k]
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(order) + 1):
            candidate = order[:pos] + [job] + order[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        order = order[:best_pos] + [job] + order[best_pos:]
    
    return order


def _local_search(processing_times, initial_order, max_iterations=1000):
    """Local search using 2-opt moves (adjacent swaps and insertions)."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 2:
        return initial_order
    
    best_order = list(initial_order)
    best_ms = _makespan(processing_times, best_order)
    
    iteration = 0
    improved = True
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try all 2-opt moves (insertion moves)
        for i in range(n_jobs):
            for j in range(n_jobs):
                if i == j:
                    continue
                
                # Remove job at position i and insert at position j
                candidate = list(best_order)
                job_to_move = candidate.pop(i)
                # Adjust index if i < j
                if i < j:
                    j -= 1
                candidate.insert(j, job_to_move)
                
                ms = _makespan(processing_times, candidate)
                if ms < best_ms - 1e-10:
                    best_order = candidate
                    best_ms = ms
                    improved = True
    
    return best_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using NEH + local search.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Generate multiple starting solutions
    candidates = []
    
    # 1. NEH rule
    neh_order = _neh_rule(processing_times)
    candidates.append(neh_order)
    
    # 2. Longest processing time first
    lpt_order = list(np.argsort(-processing_times.sum(axis=1)))
    candidates.append(lpt_order)
    
    # 3. Shortest processing time first
    spt_order = list(np.argsort(processing_times.sum(axis=1)))
    candidates.append(spt_order)
    
    # 4. Random restarts (a few)
    rng = np.random.RandomState(42)
    for _ in range(3):
        random_order = list(rng.permutation(n_jobs))
        candidates.append(random_order)
    
    # Apply local search to each candidate and keep the best
    best_order = None
    best_ms = float('inf')
    
    for candidate in candidates:
        ls_order = _local_search(processing_times, candidate, max_iterations=500)
        ms = _makespan(processing_times, ls_order)
        if ms < best_ms:
            best_ms = ms
            best_order = ls_order
    
    return best_order if best_order is not None else list(range(n_jobs))
# EVOLVE-BLOCK-END
