"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, pt):
    """Compute makespan for a given permutation."""
    n_jobs, n_machines = pt.shape
    if n_jobs == 0:
        return 0.0
    # Compute cumulative completion times
    # For each job in order, the completion time on machine k is:
    # C[j][k] = max(C[j][k-1], C[j-1][k]) + pt[j][k]
    # We only need the last machine's completion time for the last job.
    # But to be efficient, we can track the completion time per machine.
    # Actually, the standard way:
    # Let C[k] be the completion time of the current job on machine k.
    # For each job j in perm:
    #   prev = 0
    #   for k in range(n_machines):
    #       C[k] = max(C[k], prev) + pt[j][k]
    #       prev = C[k]
    # Makespan = C[-1] after all jobs.
    
    # This is O(n_jobs * n_machines) per makespan evaluation.
    C = np.zeros(n_machines, dtype=float)
    for j in perm:
        prev = 0.0
        for k in range(n_machines):
            if C[k] > prev:
                C[k] += pt[j, k]
            else:
                C[k] = prev + pt[j, k]
            prev = C[k]
    return C[-1]


def _neh_initial(pt):
    """NEH heuristic: sort by decreasing total processing time, insert each job in best position."""
    n_jobs, n_machines = pt.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    total_times = pt.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        # Try inserting j at each position in current seq
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [j] + seq[pos:]
            ms = _makespan(new_seq, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    
    return seq


def _local_search_insertion(pt, seq, max_iters=100, time_limit=10.0):
    """Local search with insertion moves (I1)."""
    n_jobs = len(seq)
    if n_jobs <= 2:
        return seq
    
    best_seq = list(seq)
    best_ms = _makespan(best_seq, pt)
    start_time = time.time()
    iters = 0
    
    while iters < max_iters:
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            job = best_seq[i]
            # Remove job
            new_seq = best_seq[:i] + best_seq[i+1:]
            # Try re-inserting at all positions
            for pos in range(n_jobs - 1):
                if pos == i or pos == i - 1:
                    continue
                trial_seq = new_seq[:pos] + [job] + new_seq[pos:]
                ms = _makespan(trial_seq, pt)
                if ms < best_ms - 1e-9:
                    best_seq = trial_seq
                    best_ms = ms
                    improved = True
                    # Restart the outer loop from the new position
                    i = pos
                    break
            if improved:
                break
        iters += 1
        if not improved:
            break
    
    return best_seq


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using:
    1. NEH heuristic for initial construction.
    2. Local search with insertion moves to refine.
    
    Uses a deadline guard to stay within the 120s evaluation limit.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Stage 1: NEH initial solution
    seq = _neh_initial(processing_times)
    
    if time.time() > deadline:
        return seq
    
    # Stage 2: Local search with insertion moves
    # Use a time limit for local search
    remaining = max(1.0, deadline - time.time())
    seq = _local_search_insertion(processing_times, seq, max_iters=200, time_limit=remaining)
    
    if time.time() > deadline:
        return seq
    
    # Stage 3: Try a few more refinement passes if time permits
    # Re-run local search with potentially more iterations
    remaining = max(0.5, deadline - time.time())
    if remaining > 2.0:
        seq = _local_search_insertion(processing_times, seq, max_iters=500, time_limit=remaining)
    
    return seq
# EVOLVE-BLOCK-END
