"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation."""
    n_machines = processing_times.shape[1]
    completion = np.zeros(n_machines)
    for job in perm:
        for m in range(n_machines):
            completion[m] = max(completion[m], completion[m - 1]) + processing_times[job, m] if m > 0 else processing_times[job, m]
    return completion[-1]


def _neh_construct(processing_times):
    """NEH heuristic: sort by total processing time, insert each job at best position."""
    n_jobs, n_machines = processing_times.shape
    total_times = processing_times.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [j] + seq[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq.insert(best_pos, j)
    return seq


def _calc_makespan(perm, processing_times):
    """Fast makespan calculation."""
    n_machines = processing_times.shape[1]
    completion = np.zeros(n_machines)
    for job in perm:
        for m in range(n_machines):
            if m == 0:
                completion[m] = processing_times[job, m]
            else:
                completion[m] = max(completion[m], completion[m-1]) + processing_times[job, m]
    return completion[-1]


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using:
    1. NEH heuristic for initial construction.
    2. Local search with insertion moves (I1) to refine.
    3. Deadline guard to ensure completion within time limit.
    """
    start_time = time.time()
    time_limit = 120.0
    safety_margin = 5.0
    deadline = start_time + time_limit - safety_margin
    
    n_jobs, n_machines = processing_times.shape
    
    # Stage 1: NEH initial construction
    seq = _neh_construct(processing_times)
    best_seq = seq.copy()
    best_ms = _calc_makespan(best_seq, processing_times)
    
    # Stage 2: Local search with insertion moves
    improved = True
    while improved:
        improved = False
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            # Remove job at position i
            job = seq[i]
            seq_without = seq[:i] + seq[i+1:]
            # Try inserting at all other positions
            for j in range(n_jobs - 1):
                if time.time() > deadline:
                    break
                if j == i or j == i - 1:
                    continue
                candidate = seq_without[:j] + [job] + seq_without[j:]
                ms = _calc_makespan(candidate, processing_times)
                if ms < best_ms:
                    best_ms = ms
                    best_seq = candidate.copy()
                    seq = candidate.copy()
                    improved = True
        if time.time() > deadline:
            break
    
    return list(best_seq)
# EVOLVE-BLOCK-END
