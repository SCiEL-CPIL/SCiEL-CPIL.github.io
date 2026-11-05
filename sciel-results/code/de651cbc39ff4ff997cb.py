"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, processing_times):
    """
    Calculate makespan for a given permutation.
    perm: list of job indices
    processing_times: np.ndarray [n_jobs, n_machines]
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0 or len(perm) == 0:
        return 0.0
    
    machine_times = [0.0] * n_machines
    max_completion = 0.0
    
    for job in perm:
        p_row = processing_times[job]
        current_time = 0.0
        for m in range(n_machines):
            # Machine m is free at machine_times[m]
            # Job is ready at current_time
            if current_time > machine_times[m]:
                start = current_time
            else:
                start = machine_times[m]
            end = start + p_row[m]
            current_time = end
            machine_times[m] = end
            if end > max_completion:
                max_completion = end
                
    return max_completion

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))
        
    # Simulated Annealing Approach
    # 1. Initial Solution: Random permutation
    current_perm = list(range(n_jobs))
    random.shuffle(current_perm)
    
    current_ms = _makespan(current_perm, processing_times)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    start_time = time.time()
    time_limit = 115.0
    
    # SA Parameters
    total_sum = np.sum(processing_times)
    T = total_sum * 0.1
    T_min = 1e-6
    alpha = 0.995
    
    while (time.time() - start_time) < time_limit:
        # Generate neighbor: Insertion move
        i = random.randint(0, n_jobs - 1)
        job = current_perm[i]
        current_perm.pop(i)
        
        # Choose insertion point in the remaining list (size n_jobs - 1)
        # Valid indices: 0 to n_jobs - 1
        j = random.randint(0, n_jobs - 1)
        
        # If j == i, it's the same position (conceptually)
        if j == i:
            current_perm.insert(j, job)
            continue
            
        current_perm.insert(j, job)
        
        new_ms = _makespan(current_perm, processing_times)
        delta = new_ms - current_ms
        
        # Acceptance criteria
        if delta < 0 or (T > T_min and random.random() < np.exp(-delta / T)):
            current_ms = new_ms
            if new_ms < best_ms:
                best_ms = new_ms
                best_perm = current_perm[:]
        else:
            # Revert
            current_perm.pop(j)
            current_perm.insert(i, job)
            
        # Cooling
        T *= alpha
        
    return best_perm
# EVOLVE-BLOCK-END
