"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using Simulated Annealing.
    This approach uses a stochastic global search strategy.
    """
    n_jobs, n_machines = processing_times.shape
    
    # 1. Initial Solution (NEH)
    total_times = processing_times.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [j] + seq[pos:]
            makespan = _calc_makespan(processing_times, candidate, n_machines)
            if makespan < best_makespan:
                best_makespan = makespan
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    
    current_seq = seq
    current_cost = _calc_makespan(processing_times, current_seq, n_machines)
    best_seq = current_seq.copy()
    best_cost = current_cost
    
    # 2. Simulated Annealing Parameters
    # Tune start_temp based on average processing time to scale acceptance
    avg_proc = processing_times.mean()
    start_temp = 10.0 * avg_proc
    end_temp = 0.01 * avg_proc
    cooling_rate = 0.999
    time_limit = 110.0  # seconds
    
    start_time = time.time()
    temp = start_temp
    
    while temp > end_temp:
        if time.time() - start_time > time_limit:
            break
            
        # Generate a neighbor
        neighbor_seq = current_seq.copy()
        move_type = random.random()
        
        if move_type < 0.5:
            # Swap move
            i, j = random.sample(range(n_jobs), 2)
            neighbor_seq[i], neighbor_seq[j] = neighbor_seq[j], neighbor_seq[i]
        else:
            # Insertion move
            i, j = random.sample(range(n_jobs), 2)
            job = neighbor_seq[i]
            neighbor_seq.pop(i)
            # Adjust j if i < j
            if i < j:
                j -= 1
            neighbor_seq.insert(j, job)
        
        neighbor_cost = _calc_makespan(processing_times, neighbor_seq, n_machines)
        
        # Acceptance Criterion
        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < np.exp(-delta / temp):
            current_seq = neighbor_seq
            current_cost = neighbor_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_seq = current_seq.copy()
        
        temp *= cooling_rate
    
    return best_seq

def _calc_makespan(processing_times, seq, n_machines):
    """Calculate the makespan for a given job sequence."""
    completion_times = np.zeros(n_machines)
    
    for job_idx in seq:
        for m in range(n_machines):
            if m == 0:
                completion_times[m] += processing_times[job_idx, m]
            else:
                completion_times[m] = max(completion_times[m], completion_times[m-1]) + processing_times[job_idx, m]
    
    return completion_times[-1]
# EVOLVE-BLOCK-END
