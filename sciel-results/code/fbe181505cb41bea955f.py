"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """Compute the makespan for a given permutation."""
    n_jobs, n_machines = processing_times.shape
    # For each machine, the completion time is the max over jobs of (sum of processing times on that machine up to job + sum of processing times on previous machines for that job)
    # More efficiently: C[j][m] = max(C[j][m-1], C[j-1][m]) + p[j][m]
    # We can compute this in O(n_jobs * n_machines)
    # Using vectorized approach:
    # For a given permutation, we can compute the makespan as follows:
    # C_m = max over jobs in order of (cumulative sum of p[j][m] for jobs up to j in order + cumulative sum of p[j][m-1] for jobs up to j in order)
    # This is complex to vectorize. Let's use a simple loop which is O(n*m).
    
    # Actually, for small n and m, a loop is fine. But let's try to be efficient.
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + p[perm[j]][m]
    # We only need the final row.
    
    # Let's use a 1D array for the previous machine's completion times
    # prev_machine_times[m] = completion time of the last job on machine m
    # No, that's not right. We need the completion time of each job on each machine.
    
    # Let's stick with the standard O(n*m) algorithm
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + p[j][m]
    # We can compute this row by row.
    
    # For efficiency, let's pre-extract the processing times for the jobs in the given order
    # p_ordered = processing_times[perm]
    # Then C[j][m] = max(C[j][m-1], C[j-1][m]) + p_ordered[j][m]
    
    # We only need the last row, so we can keep a 1D array of the previous row
    # prev_row[m] = C[j-1][m]
    # current_row[m] = max(prev_row[m], current_row[m-1]) + p_ordered[j][m]
    
    # Let's implement this
    p_ordered = processing_times[perm]
    n = len(perm)
    m = n_machines
    
    if n == 0:
        return 0.0
    
    # Initialize the completion times for the first job
    # C[0][m] = sum(p_ordered[0][:m+1])
    # We can compute this as a cumulative sum
    prev_row = np.cumsum(p_ordered[0])
    
    for j in range(1, n):
        # current_row[m] = max(prev_row[m], current_row[m-1]) + p_ordered[j][m]
        # We compute this in-place on prev_row, but we need the previous value of prev_row[m]
        # Let's use a temporary variable for the current row's previous machine
        current_prev = 0.0
        for m_idx in range(m):
            # max(prev_row[m_idx], current_prev) + p_ordered[j][m_idx]
            val = max(prev_row[m_idx], current_prev) + p_ordered[j][m_idx]
            prev_row[m_idx] = val
            current_prev = val
    
    return prev_row[-1]


def _neh_heuristic(processing_times):
    """Construct initial sequence using NEH heuristic."""
    n_jobs, n_machines = processing_times.shape
    total_times = processing_times.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [j] + seq[pos:]
            ms = _makespan(new_seq, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    
    return seq


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using NEH + local search.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10 seconds safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Stage 1: NEH initial solution
    best_seq = _neh_heuristic(processing_times)
    best_ms = _makespan(best_seq, processing_times)
    
    # Stage 2: Local search with insertion moves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            job = best_seq[i]
            for j in range(n_jobs):
                if i == j:
                    continue
                if time.time() > deadline:
                    break
                # Try moving job from position i to position j
                if j > i:
                    new_seq = best_seq[:i] + best_seq[i+1:j+1] + [job] + best_seq[j+1:]
                else:
                    new_seq = best_seq[:j] + [job] + best_seq[j:i] + best_seq[i+1:]
                
                new_ms = _makespan(new_seq, processing_times)
                if new_ms < best_ms - 1e-9:
                    best_ms = new_ms
                    best_seq = new_seq
                    improved = True
                    # After a move, restart the inner loop to re-evaluate from the new position
                    # Actually, it's better to break and restart the outer loop
                    break
            if improved:
                break
    
    return list(best_seq)
# EVOLVE-BLOCK-END
