"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using:
    1. Vierling's Heuristic for initial construction.
    2. Simulated Annealing with adjacent swap moves for optimization.
    
    This approach differs from NEH+Insertion by using a different construction
    heuristic and a probabilistic metaheuristic (SA) to escape local minima.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
        
    # --- Stage 1: Vierling's Heuristic ---
    seq = _vierling(processing_times)
    current_makespan = _calc_makespan(processing_times, seq, n_machines)
    best_seq = seq.copy()
    best_makespan = current_makespan
    
    # --- Stage 2: Simulated Annealing ---
    # Time limit to ensure we finish within 120s total
    start_time = time.time()
    max_time = 20.0  # 20 seconds for SA
    
    # Initial temperature based on problem size and makespan
    T = current_makespan / 10.0
    if T < 1: T = 1.0
    cooling_rate = 0.995
    
    # Use a random seed for reproducibility
    rng = np.random.default_rng(seed=42)
    
    while T > 1e-4 and (time.time() - start_time) < max_time:
        # Generate a random adjacent swap
        i = rng.integers(0, n_jobs - 1)
        j = i + 1
        
        # Perform swap
        seq[i], seq[j] = seq[j], seq[i]
        new_makespan = _calc_makespan(processing_times, seq, n_machines)
        
        delta = new_makespan - current_makespan
        
        if delta < 0:
            # Better solution
            current_makespan = new_makespan
            if new_makespan < best_makespan:
                best_makespan = new_makespan
                best_seq = seq.copy()
        else:
            # Worse solution, accept with probability exp(-delta/T)
            if rng.random() < np.exp(-delta / T):
                current_makespan = new_makespan
            else:
                # Reject move, swap back
                seq[i], seq[j] = seq[j], seq[i]
        
        T *= cooling_rate

    return best_seq

def _vierling(processing_times):
    """
    Implement Vierling's heuristic for flow-shop problem.
    1. Select job with max processing time on machine 1.
    2. Select job with min processing time on machine M.
    3. Place them at the ends.
    4. Repeat for remaining jobs.
    """
    n_jobs, n_machines = processing_times.shape
    remaining_jobs = list(range(n_jobs))
    seq = [None] * n_jobs
    
    left_idx = 0
    right_idx = n_jobs - 1
    
    while remaining_jobs:
        # Select job with max processing time on machine 0
        max_job = max(remaining_jobs, key=lambda j: processing_times[j, 0])
        
        # Select job with min processing time on machine M-1
        min_job = min(remaining_jobs, key=lambda j: processing_times[j, n_machines-1])
        
        if max_job == min_job:
            seq[left_idx] = max_job
            remaining_jobs.remove(max_job)
            left_idx += 1
            if left_idx > right_idx:
                break
        else:
            seq[left_idx] = max_job
            seq[right_idx] = min_job
            remaining_jobs.remove(max_job)
            remaining_jobs.remove(min_job)
            left_idx += 1
            right_idx -= 1
            
    return seq

def _calc_makespan(processing_times, seq, n_machines):
    """Calculate the makespan for a given job sequence."""
    completion_times = np.zeros(n_machines)
    pt = processing_times
    for job_idx in seq:
        row = pt[job_idx]
        completion_times[0] += row[0]
        for m in range(1, n_machines):
            completion_times[m] = max(completion_times[m], completion_times[m-1]) + row[m]
    return completion_times[-1]
# EVOLVE-BLOCK-END
