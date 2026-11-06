import numpy as np
import time
import random

def _calc_makespan(processing_times, perm):
    """Calculate makespan for a given permutation."""
    n_machines = processing_times.shape[1]
    completion = np.zeros(n_machines)
    for j in perm:
        job_times = processing_times[j]
        for m in range(n_machines):
            if m == 0:
                start_time = completion[0]
            else:
                start_time = max(completion[m], completion[m-1])
            completion[m] = start_time + job_times[m]
    return completion[-1]

def _neh_heuristic(processing_times):
    """NEH heuristic for initial solution."""
    n_jobs, n_machines = processing_times.shape
    total_times = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_times)
    
    seq = [sorted_jobs[0]]
    
    for j in sorted_jobs[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [j] + seq[pos:]
            ms = _calc_makespan(processing_times, new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        seq.insert(best_pos, j)
    
    return seq

def _local_search(processing_times, initial_seq, deadline):
    """Local search with insertion moves."""
    n_jobs = len(initial_seq)
    seq = list(initial_seq)
    best_ms = _calc_makespan(processing_times, seq)
    
    while time.time() < deadline:
        improved = False
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            job = seq[i]
            # Remove job at position i
            seq_no_job = seq[:i] + seq[i+1:]
            
            # Try inserting at each position
            for j in range(n_jobs - 1):
                if j == i:
                    continue
                new_seq = seq_no_job[:j] + [job] + seq_no_job[j:]
                ms = _calc_makespan(processing_times, new_seq)
                
                if ms < best_ms:
                    best_ms = ms
                    seq = new_seq
                    improved = True
                    break
        
        if not improved:
            break
    
    return seq, best_ms

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    # 1. NEH Construction
    seq = _neh_heuristic(processing_times)
    
    # 2. Local search if time permits
    if time.time() < deadline - 5.0:
        seq, best_ms = _local_search(processing_times, seq, deadline)
    
    return list(seq)
