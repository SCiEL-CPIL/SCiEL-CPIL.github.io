"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import permutations


def _makespan(perm, processing_times):
    """Calculate makespan for a given permutation."""
    n_jobs, n_machines = processing_times.shape
    completion_times = np.zeros(n_jobs)
    for job in perm:
        for m in range(n_machines):
            if m == 0:
                completion_times[job] = processing_times[job, m]
            else:
                completion_times[job] = completion_times[job] + processing_times[job, m]
    return completion_times.max()


def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    # If small enough, try exact search with branch and bound or simple permutation
    # For very small instances, we can use a more sophisticated approach
    
    # Start with a good heuristic: NEH (Nawaz-Enscore-Ham)
    # 1. Calculate total processing time for each job
    totals = processing_times.sum(axis=1)
    
    # 2. Sort jobs in decreasing order of total processing time
    initial_perm = np.argsort(-totals).tolist()
    
    # 3. Build the schedule using NEH insertion
    # Actually, let's implement a proper NEH heuristic
    # Sort jobs by total processing time descending
    sorted_jobs = np.argsort(-totals)
    
    # Build schedule by inserting each job in the best position
    current_perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_perm) + 1):
            # Try inserting job at position pos
            test_perm = current_perm[:pos] + [job] + current_perm[pos:]
            
            # Calculate makespan for this partial schedule
            # We only need to calculate up to the inserted job and subsequent jobs
            # For efficiency, we can just calculate the full makespan for the partial schedule
            if len(test_perm) == 0:
                continue
                
            # Calculate makespan for the partial schedule
            comp = np.zeros(n_jobs)
            for j in test_perm:
                for m in range(n_machines):
                    if m == 0:
                        comp[j] = processing_times[j, m]
                    else:
                        comp[j] = comp[j] + processing_times[j, m]
            
            ms = comp.max()
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_perm = current_perm[:best_pos] + [list(sorted_jobs)[list(sorted_jobs).index(job)]] + current_perm[best_pos:]
    
    # Actually, let me redo this more carefully
    # NEH heuristic:
    # 1. Sort jobs in non-increasing order of total processing time
    # 2. Schedule the first job
    # 3. For each subsequent job, insert it in the position that minimizes makespan
    
    sorted_jobs_list = sorted(range(n_jobs), key=lambda j: -totals[j])
    
    current_perm = [sorted_jobs_list[0]]
    
    for i in range(1, n_jobs):
        job = sorted_jobs_list[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_perm) + 1):
            test_perm = current_perm[:pos] + [job] + current_perm[pos:]
            
            # Calculate makespan for this permutation
            comp = np.zeros(n_jobs)
            for j in test_perm:
                for m in range(n_machines):
                    if m == 0:
                        comp[j] = processing_times[j, m]
                    else:
                        comp[j] = comp[j] + processing_times[j, m]
            
            ms = comp.max()
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
    
    # Now try to improve with a simple local search: pairwise swaps
    best_perm = current_perm[:]
    best_ms = _makespan(best_perm, processing_times)
    
    # If the instance is small, do more extensive search
    if n_jobs <= 8:
        # Try all permutations for very small instances
        best_perm = None
        best_ms = float('inf')
        for perm in permutations(range(n_jobs)):
            ms = _makespan(list(perm), processing_times)
            if ms < best_ms:
                best_ms = ms
                best_perm = list(perm)
        return best_perm
    
    # Local search: try swapping pairs
    improved = True
    while improved:
        improved = False
        for i in range(len(best_perm)):
            for j in range(i + 1, len(best_perm)):
                # Try swapping i and j
                test_perm = best_perm[:]
                test_perm[i], test_perm[j] = test_perm[j], test_perm[i]
                ms = _makespan(test_perm, processing_times)
                if ms < best_ms:
                    best_perm = test_perm
                    best_ms = ms
                    improved = True
    
    return best_perm
# EVOLVE-BLOCK-END
