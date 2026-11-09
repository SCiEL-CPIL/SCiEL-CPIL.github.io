"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(processing_times, order):
    """Compute makespan for a given job order using the recurrence efficiently."""
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    # c[j] holds C[i][j] for the current job i
    c = np.zeros(n_machines)
    for job_idx in order:
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            if j == 0:
                c[j] = c[j] + p_row[j]
            else:
                c[j] = max(c[j], c[j-1]) + p_row[j]
    return c[-1]


def _insertion_makespan(processing_times, order, job_idx, pos):
    """Compute makespan after inserting job_idx at position pos in order."""
    # Create new order
    new_order = list(order[:pos]) + [job_idx] + list(order[pos:])
    return _makespan(processing_times, new_order)


def _neh_rule(processing_times):
    """NEH (Newell, Eberhart, and Heskett) heuristic."""
    n_jobs = processing_times.shape[0]
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Insert each job into the best position
    final_order = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(final_order) + 1):
            new_order = final_order[:pos] + [job] + final_order[pos:]
            ms = _makespan(processing_times, new_order)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        final_order = final_order[:best_pos] + [job] + final_order[best_pos:]
    
    return final_order


def _local_search(processing_times, order, time_limit=10.0):
    """First-improvement local search with 2-opt style moves."""
    start_time = time.time()
    n_jobs = len(order)
    current_ms = _makespan(processing_times, order)
    improved = True
    
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(n_jobs - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n_jobs):
                if time.time() - start_time > time_limit:
                    break
                # Try swapping jobs at positions i and j
                new_order = list(order)
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_ms = _makespan(processing_times, new_order)
                if new_ms < current_ms:
                    order = new_order
                    current_ms = new_ms
                    improved = True
                    break  # First improvement: restart scan
            if improved:
                break
    
    return order, current_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_jobs = processing_times.shape[0]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Phase 1: NEH heuristic for initial solution
    order = _neh_rule(processing_times)
    current_ms = _makespan(processing_times, order)
    
    # Phase 2: Local search with time limit
    if time.time() < deadline - 5.0:
        order, current_ms = _local_search(processing_times, order, time_limit=10.0)
    
    # Phase 3: If time remains, try multiple random restarts with NEH + LS
    if time.time() < deadline - 10.0:
        best_order = list(order)
        best_ms = current_ms
        
        n_restarts = 0
        while time.time() < deadline - 5.0:
            n_restarts += 1
            # Generate a perturbed initial order
            if n_restarts % 2 == 0:
                # Random order
                restart_order = list(range(n_jobs))
                random.shuffle(restart_order)
            else:
                # NEH with random tie-breaking
                totals = processing_times.sum(axis=1)
                # Add small random noise to break ties
                noisy_totals = totals + np.random.random(n_jobs) * 0.01
                restart_order = list(np.argsort(-noisy_totals))
            
            # Apply NEH insertion
            final_order = [restart_order[0]]
            for i in range(1, n_jobs):
                job = restart_order[i]
                best_pos = 0
                best_ms_local = float('inf')
                for pos in range(len(final_order) + 1):
                    new_order = final_order[:pos] + [job] + final_order[pos:]
                    ms = _makespan(processing_times, new_order)
                    if ms < best_ms_local:
                        best_ms_local = ms
                        best_pos = pos
                final_order = final_order[:best_pos] + [job] + final_order[best_pos:]
            
            # Quick local search
            if time.time() < deadline - 5.0:
                final_order, final_ms = _local_search(processing_times, final_order, time_limit=5.0)
            else:
                final_ms = _makespan(processing_times, final_order)
            
            if final_ms < best_ms:
                best_ms = final_ms
                best_order = list(final_order)
    
    return best_order
# EVOLVE-BLOCK-END
