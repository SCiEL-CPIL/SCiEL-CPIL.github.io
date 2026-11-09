"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    # completion[m] = completion time of current job on machine m
    # We process jobs in 'order', tracking the completion time on each machine
    # completion[i] for machine i after processing the current job
    completion = np.zeros(n_machines)
    for job in order:
        for m in range(n_machines):
            if m == 0:
                completion[m] += processing_times[job, m]
            else:
                completion[m] = max(completion[m - 1], completion[m]) + processing_times[job, m]
    return completion[-1]

def _makespan_vectorized(processing_times, order):
    """Vectorized makespan computation for speed."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    # completion[m] after each job
    # Use a loop over jobs but vectorize over machines where possible
    completion = np.zeros(n_machines)
    for job in order:
        p = processing_times[job]
        # For machine 0: completion[0] += p[0]
        # For machine m>0: completion[m] = max(completion[m-1], completion[m]) + p[m]
        # This is inherently sequential across machines, but we can do it in a loop
        # Actually, let's just do the straightforward loop; for small n_machines it's fine
        for m in range(n_machines):
            if m == 0:
                completion[m] += p[m]
            else:
                completion[m] = max(completion[m - 1], completion[m]) + p[m]
    return completion[-1]

def _neh_heuristic(processing_times):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    # Sort jobs by total processing time in descending order
    total_times = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-total_times))
    
    # Build the schedule by inserting each job in the best position
    order = []
    for job in sorted_jobs:
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting at each position
        for pos in range(len(order) + 1):
            candidate_order = order[:pos] + [job] + order[pos:]
            ms = _makespan_vectorized(processing_times, candidate_order)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        order = order[:best_pos] + [job] + order[best_pos:]
    
    return order

def _local_search(processing_times, initial_order, time_limit=5.0):
    """First-improvement local search with 2-opt moves."""
    n_jobs = len(initial_order)
    if n_jobs <= 1:
        return list(initial_order)
    
    best_order = list(initial_order)
    best_makespan = _makespan_vectorized(processing_times, best_order)
    start_time = time.time()
    
    improved = True
    while improved:
        improved = False
        # Try all pairs of swaps
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n_jobs):
                # Swap i and j
                new_order = best_order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
                ms = _makespan_vectorized(processing_times, new_order)
                if ms < best_makespan - 1e-12:
                    best_order = new_order
                    best_makespan = ms
                    improved = True
                    break
            if time.time() - start_time > time_limit:
                break
    
    return best_order

def _simulated_annealing(processing_times, initial_order, time_limit=10.0):
    """Simulated annealing for flow shop scheduling."""
    n_jobs = len(initial_order)
    if n_jobs <= 1:
        return list(initial_order)
    
    current_order = list(initial_order)
    current_makespan = _makespan_vectorized(processing_times, current_order)
    best_order = list(current_order)
    best_makespan = current_makespan
    
    start_time = time.time()
    initial_temp = max(current_makespan * 0.1, 1.0)
    cooling_rate = 0.99
    
    temp = initial_temp
    rng = np.random.default_rng(42)
    
    while temp > 1e-6 and time.time() - start_time < time_limit:
        # Generate a neighbor by swapping two random jobs
        i = rng.integers(0, n_jobs)
        j = rng.integers(0, n_jobs)
        if i == j:
            continue
        
        new_order = current_order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        new_makespan = _makespan_vectorized(processing_times, new_order)
        
        delta = new_makespan - current_makespan
        
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            current_order = new_order
            current_makespan = new_makespan
            
            if new_makespan < best_makespan:
                best_order = list(new_order)
                best_makespan = new_makespan
        
        temp *= cooling_rate
    
    return best_order

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Uses NEH heuristic followed by local search and simulated annealing.
    """
    processing_times = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Step 1: NEH heuristic
    neh_order = _neh_heuristic(processing_times)
    neh_makespan = _makespan_vectorized(processing_times, neh_order)
    
    # Step 2: Local search from NEH
    ls_order = _local_search(processing_times, neh_order, time_limit=10.0)
    ls_makespan = _makespan_vectorized(processing_times, ls_order)
    
    best_order = ls_order if ls_makespan < neh_makespan else neh_order
    best_makespan = min(ls_makespan, neh_makespan)
    
    # Step 3: Simulated annealing from best so far
    elapsed = time.time() - start_time
    remaining = deadline - elapsed - 5.0  # 5s safety margin
    if remaining > 5.0:
        sa_order = _simulated_annealing(processing_times, best_order, time_limit=remaining)
        sa_makespan = _makespan_vectorized(processing_times, sa_order)
        if sa_makespan < best_makespan:
            best_order = sa_order
            best_makespan = sa_makespan
    
    # Step 4: Final local search polish if time permits
    elapsed = time.time() - start_time
    remaining = deadline - elapsed - 2.0
    if remaining > 2.0:
        polished_order = _local_search(processing_times, best_order, time_limit=remaining)
        polished_makespan = _makespan_vectorized(processing_times, polished_order)
        if polished_makespan < best_makespan:
            best_order = polished_order
            best_makespan = polished_makespan
    
    return list(best_order)
# EVOLVE-BLOCK-END
