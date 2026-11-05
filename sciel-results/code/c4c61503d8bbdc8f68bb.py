"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(perm, processing_times):
    """
    Efficiently calculate makespan for a given permutation.
    perm: list of job indices
    processing_times: np.ndarray [n_jobs, n_machines]
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    last_machine_time = [0.0] * n_machines
    max_completion = 0.0
    
    for job in perm:
        current_job_time = 0.0
        # Unroll loop for small M? No, keep general.
        for m in range(n_machines):
            start_time = max(current_job_time, last_machine_time[m])
            end_time = start_time + processing_times[job, m]
            current_job_time = end_time
            last_machine_time[m] = end_time
            if end_time > max_completion:
                max_completion = end_time
    return max_completion

def _neh_initial_solution(processing_times):
    """
    NEH (Nawaz, Enscore, Ham) heuristic.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    current_seq = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current_seq = current_seq[:best_pos] + [job] + current_seq[best_pos:]
        
    return current_seq

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))
        
    # 1. Construct initial solution using NEH
    current_perm = _neh_initial_solution(processing_times)
    current_ms = _makespan(current_perm, processing_times)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    start_time = time.time()
    time_limit = 115.0
    
    # 2. Local Search: 2-Opt (Swap) with Best Improvement
    # To prevent timeout on large instances, we limit the number of passes 
    # or the size of the neighborhood if N is large.
    
    # For N > 100, exhaustive 2-opt is too slow. 
    # We will perform a limited number of best-improvement passes.
    max_passes = 50
    pass_count = 0
    
    while pass_count < max_passes and (time.time() - start_time) < time_limit:
        best_swap = None
        best_swap_ms = current_ms
        
        n = len(current_perm)
        
        # If N is very large, restrict search to a subset or use random sampling?
        # For now, assume N is manageable. If N > 100, we might just do 1 pass.
        # Let's add a check: if N > 80, we do a faster local search (e.g., only check neighbors)
        
        if n > 80:
             # Faster local search: only check adjacent swaps and a few random swaps?
             # Or just skip local search if too big?
             # NEH is already good. Let's do a simple hill climb on adjacent swaps.
             improved = True
             while improved and (time.time() - start_time) < time_limit:
                 improved = False
                 for i in range(n - 1):
                     new_perm = current_perm[:]
                     new_perm[i], new_perm[i+1] = current_perm[i+1], current_perm[i]
                     new_ms = _makespan(new_perm, processing_times)
                     if new_ms < current_ms:
                         current_perm = new_perm
                         current_ms = new_ms
                         improved = True
                         if current_ms < best_ms:
                             best_ms = current_ms
                             best_perm = current_perm[:]
             break
        else:
            # Standard Best Improvement 2-Opt
            for i in range(n):
                for j in range(i + 1, n):
                    new_perm = current_perm[:]
                    new_perm[i], new_perm[j] = current_perm[j], new_perm[i]
                    
                    new_ms = _makespan(new_perm, processing_times)
                    
                    if new_ms < best_swap_ms:
                        best_swap_ms = new_ms
                        best_swap = (i, j)
                        
            if best_swap is not None:
                i, j = best_swap
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                current_ms = best_swap_ms
                pass_count += 1
                
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_perm = current_perm[:]
            else:
                # Local optimum
                break
    return best_perm
# EVOLVE-BLOCK-END
