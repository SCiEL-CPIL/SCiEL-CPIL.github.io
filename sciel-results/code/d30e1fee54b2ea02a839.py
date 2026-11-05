# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """Compute makespan for a given job order on a flow shop."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    # completion[j][m] = completion time of job j on machine m
    # We can compute this incrementally
    # For a given order, we track the completion time on the last machine for each job
    # and the completion time on each machine.
    
    # completion[job] = time when this job finishes on the last machine
    # machine_completion[m] = time when machine m becomes free
    # But we need to be careful: for job j in order, on machine m,
    # start = max(completion of previous job on machine m, completion of this job on previous machine)
    
    # Let's use a 1D array for machine completion times and a scalar for the current job's completion on previous machine.
    machine_completion = np.zeros(n_machines)
    total = 0.0
    for job in order:
        prev_completion = 0.0  # completion time of this job on the previous machine
        for m in range(n_machines):
            # start time on machine m for this job
            start = max(machine_completion[m], prev_completion)
            end = start + processing_times[job, m]
            machine_completion[m] = end
            prev_completion = end
        total = prev_completion
    return total

def _neh_heuristic(processing_times):
    """NEH (Nawaz-Enscore-Ham) heuristic for flow shop."""
    n_jobs, n_machines = processing_times.shape
    # Sort jobs in descending order of total processing time
    total_times = processing_times.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    # Insert each job into the best position
    schedule = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(schedule) + 1):
            new_schedule = schedule[:pos] + [job] + schedule[pos:]
            ms = _makespan(processing_times, new_schedule)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        schedule.insert(best_pos, job)
    
    return schedule

def _cds_heuristic(processing_times):
    """CDS (Campbell, Dudek, Smith) heuristic for flow shop."""
    n_jobs, n_machines = processing_times.shape
    total_times = processing_times.sum(axis=1)
    order = list(np.argsort(total_times))
    
    schedule = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(schedule) + 1):
            new_schedule = schedule[:pos] + [job] + schedule[pos:]
            ms = _makespan(processing_times, new_schedule)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        schedule.insert(best_pos, job)
    
    return schedule

def _two_opt(processing_times, order, max_iterations=1000, timeout=10.0):
    """First-improvement 2-opt (interchange) local search."""
    n_jobs = len(order)
    current_ms = _makespan(processing_times, order)
    best_order = list(order)
    best_ms = current_ms
    
    start_time = time.time()
    iterations = 0
    
    while iterations < max_iterations:
        if time.time() - start_time > timeout:
            break
        
        improved = False
        for i in range(n_jobs - 1):
            if time.time() - start_time > timeout:
                break
            for j in range(i + 1, n_jobs):
                # Try swapping jobs at positions i and j
                new_order = list(best_order)
                new_order[i], new_order[j] = new_order[j], new_order[i]
                ms = _makespan(processing_times, new_order)
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    best_order = new_order
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
        iterations += 1
    
    return best_order, best_ms

def _insertion_search(processing_times, order, max_iterations=500, timeout=5.0):
    """First-improvement insertion local search."""
    n_jobs = len(order)
    best_order = list(order)
    best_ms = _makespan(processing_times, best_order)
    
    start_time = time.time()
    iterations = 0
    
    while iterations < max_iterations:
        if time.time() - start_time > timeout:
            break
        
        improved = False
        for i in range(n_jobs):
            if time.time() - start_time > timeout:
                break
            job = best_order[i]
            # Remove job at position i
            remaining = best_order[:i] + best_order[i+1:]
            for j in range(len(remaining) + 1):
                # Insert job at position j
                new_order = remaining[:j] + [job] + remaining[j:]
                ms = _makespan(processing_times, new_order)
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    best_order = new_order
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
        iterations += 1
    
    return best_order, best_ms

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan on a flow shop.
    Uses multiple heuristics and local search with a time budget.
    """
    processing_times = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    deadline = time.time() + 115.0  # Safety margin: 5s before 120s limit
    
    # Generate initial solutions using different heuristics
    candidates = []
    
    # NEH
    neh_order = _neh_heuristic(processing_times)
    candidates.append(neh_order)
    
    # CDS
    cds_order = _cds_heuristic(processing_times)
    candidates.append(cds_order)
    
    # Longest processing time first (LPT)
    lpt_order = list(np.argsort(-processing_times.sum(axis=1)))
    candidates.append(lpt_order)
    
    # Random permutations (a few)
    for _ in range(3):
        rand_order = list(range(n_jobs))
        random.shuffle(rand_order)
        candidates.append(rand_order)
    
    # Evaluate all candidates and keep the best
    best_order = None
    best_ms = float('inf')
    
    for order in candidates:
        ms = _makespan(processing_times, order)
        if ms < best_ms:
            best_ms = ms
            best_order = list(order)
    
    # Apply local search to the best candidate
    if time.time() < deadline - 5.0:
        best_order, best_ms = _two_opt(processing_times, best_order, timeout=min(30.0, deadline - time.time() - 1.0))
    
    if time.time() < deadline - 2.0:
        best_order, best_ms = _insertion_search(processing_times, best_order, timeout=min(15.0, deadline - time.time() - 0.5))
    
    # Final check: if we're close to deadline, return immediately
    return list(best_order)

# EVOLVE-BLOCK-END
