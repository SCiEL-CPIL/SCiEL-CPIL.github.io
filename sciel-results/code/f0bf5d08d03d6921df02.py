"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _makespan(perm, processing_times):
    """
    Efficiently calculate makespan for a given permutation.
    perm: list of job indices
    processing_times: np.ndarray [n_jobs, n_machines]
    """
    n_machines = processing_times.shape[1]
    last_machine_time = [0] * n_machines
    max_completion = 0
    
    for job in perm:
        current_job_time = 0
        for m in range(n_machines):
            start_time = max(current_job_time, last_machine_time[m])
            end_time = start_time + processing_times[job, m]
            current_job_time = end_time
            last_machine_time[m] = end_time
            if end_time > max_completion:
                max_completion = end_time
    return max_completion

def _neh_init(processing_times):
    """
    NEH heuristic initialization for flow shop.
    """
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(totals, kind='stable'))
    sorted_jobs.reverse()  # descending order of totals
    
    perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            trial = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(trial, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    return perm

def _vnd(perm, processing_times):
    """
    Variable Neighborhood Descent with insert and swap neighborhoods.
    Uses best improvement strategy for quality.
    """
    n_jobs = len(perm)
    current_ms = _makespan(perm, processing_times)
    improved = True
    
    while improved:
        improved = False
        
        # Insert neighborhood - best improvement
        best_ms = current_ms
        best_perm = perm[:]
        for i in range(n_jobs):
            job = perm[i]
            temp_perm = perm[:i] + perm[i+1:]
            for j in range(len(temp_perm) + 1):
                if j == i or j == i - 1:
                    continue
                new_perm = temp_perm[:j] + [job] + temp_perm[j:]
                new_ms = _makespan(new_perm, processing_times)
                if new_ms < best_ms:
                    best_ms = new_ms
                    best_perm = new_perm
        
        if best_ms < current_ms:
            perm = best_perm
            current_ms = best_ms
            improved = True
            continue
        
        # Swap neighborhood - best improvement
        best_ms = current_ms
        best_perm = perm[:]
        for i in range(n_jobs):
            for j in range(i + 1, n_jobs):
                new_perm = perm[:]
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_ms = _makespan(new_perm, processing_times)
                if new_ms < best_ms:
                    best_ms = new_ms
                    best_perm = new_perm
        
        if best_ms < current_ms:
            perm = best_perm
            current_ms = best_ms
            improved = True
    
    return perm, current_ms

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    random.seed(42)
    start_time = time.time()
    time_limit = 115.0
    
    # Initialize with NEH
    perm = _neh_init(processing_times)
    best_perm, best_ms = _vnd(perm, processing_times)
    
    # Iterated Local Search
    while (time.time() - start_time) < time_limit:
        # Perturbation: random insertions
        perturbed = best_perm[:]
        n_perturb = max(1, n_jobs // 5)
        for _ in range(n_perturb):
            i = random.randint(0, n_jobs - 1)
            job = perturbed[i]
            temp = perturbed[:i] + perturbed[i+1:]
            j = random.randint(0, len(temp))
            perturbed = temp[:j] + [job] + temp[j:]
        
        # Local search on perturbed solution
        local_perm, local_ms = _vnd(perturbed, processing_times)
        
        # Accept if better
        if local_ms < best_ms:
            best_perm = local_perm
            best_ms = local_ms
    
    return best_perm
# EVOLVE-BLOCK-END
