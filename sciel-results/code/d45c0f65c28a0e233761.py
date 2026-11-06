"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(pt, order):
    """Compute makespan for a given job order. pt is (n_jobs, n_machines)."""
    n_machines = pt.shape[1]
    # Cumulative completion times for each machine
    # For each machine, we compute the cumulative sum of processing times in the order
    # but with the constraint that job i on machine m starts after job i-1 finishes on machine m-1
    # Efficient computation:
    # Let C[m][j] be the completion time of job j (0-indexed in order) on machine m
    # C[0][j] = C[0][j-1] + pt[order[j]][0]
    # C[m][j] = max(C[m-1][j], C[m][j-1]) + pt[order[j]][m]
    
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    
    # Use a 1D array for the previous machine's completion times
    prev = np.zeros(n_jobs, dtype=np.float64)
    for m in range(n_machines):
        curr = np.zeros(n_jobs, dtype=np.float64)
        for j in range(n_jobs):
            job = order[j]
            if j == 0:
                curr[j] = pt[job, m] if m == 0 else max(prev[j], pt[job, m])
                # Actually for j==0, prev[0] is the completion on previous machine for first job
                # C[m][0] = max(C[m-1][0], 0) + pt[order[0]][m] = C[m-1][0] + pt[order[0]][m]
                # Wait, that's not right. C[m-1][0] is the completion of the first job on machine m-1.
                # The first job on machine m starts after it finishes on machine m-1 AND after the previous job (none) finishes on machine m.
                # So C[m][0] = C[m-1][0] + pt[order[0]][m]
                # But prev[j] stores C[m-1][j]. So curr[0] = prev[0] + pt[order[0]][m]
                # For j > 0: curr[j] = max(prev[j], curr[j-1]) + pt[order[j]][m]
                pass
        # Let me redo this more carefully
        curr = np.zeros(n_jobs, dtype=np.float64)
        for j in range(n_jobs):
            job = order[j]
            if m == 0:
                if j == 0:
                    curr[j] = pt[job, m]
                else:
                    curr[j] = curr[j-1] + pt[job, m]
            else:
                if j == 0:
                    curr[j] = prev[j] + pt[job, m]
                else:
                    curr[j] = max(prev[j], curr[j-1]) + pt[job, m]
        prev = curr
    return prev[-1]


def _makespan_fast(pt, order):
    """Faster makespan computation using numpy operations where possible."""
    n_machines = pt.shape[1]
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    
    # Extract processing times in the given order
    ordered_pt = pt[order]  # shape (n_jobs, n_machines)
    
    # For each machine, compute cumulative times
    # C[m][j] = max(C[m-1][j], C[m][j-1]) + ordered_pt[j][m]
    # This is inherently sequential along j, but we can use numpy for the max
    
    prev = np.zeros(n_jobs, dtype=np.float64)
    for m in range(n_machines):
        curr = np.empty(n_jobs, dtype=np.float64)
        col = ordered_pt[:, m]
        for j in range(n_jobs):
            if m == 0:
                if j == 0:
                    curr[j] = col[j]
                else:
                    curr[j] = curr[j-1] + col[j]
            else:
                if j == 0:
                    curr[j] = prev[j] + col[j]
                else:
                    curr[j] = max(prev[j], curr[j-1]) + col[j]
        prev = curr
    return prev[-1]


def _neh_initial(pt):
    """NEH heuristic: sort by decreasing total processing time, insert each job in best position."""
    n_jobs, n_machines = pt.shape
    total_times = pt.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    seq = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            trial = seq[:pos] + [job] + seq[pos:]
            ms = _makespan_fast(pt, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    return seq


def _best_insertion(pt, seq):
    """For each job, find the best position to insert it (removing from current position).
    Returns the best single move and the new sequence if improvement is found."""
    n_jobs = len(seq)
    best_ms = _makespan_fast(pt, seq)
    best_seq = list(seq)
    best_improvement = 0.0
    
    for i in range(n_jobs):
        job = seq[i]
        # Remove job from position i
        remaining = seq[:i] + seq[i+1:]
        for pos in range(len(remaining) + 1):
            if pos == i:
                continue  # Same position
            trial = remaining[:pos] + [job] + remaining[pos:]
            ms = _makespan_fast(pt, trial)
            if ms < best_ms - 1e-12:
                best_ms = ms
                best_seq = trial
                best_improvement = best_improvement  # track
    
    if best_improvement > 0 or best_ms < _makespan_fast(pt, seq) - 1e-12:
        return best_seq, best_ms
    return list(seq), best_ms


def _swap_improvement(pt, seq):
    """Try all swaps and find the best one."""
    n_jobs = len(seq)
    best_ms = _makespan_fast(pt, seq)
    best_seq = list(seq)
    
    for i in range(n_jobs):
        for j in range(i+1, n_jobs):
            trial = list(seq)
            trial[i], trial[j] = trial[j], trial[i]
            ms = _makespan_fast(pt, trial)
            if ms < best_ms - 1e-12:
                best_ms = ms
                best_seq = trial
    
    return best_seq, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using:
    1. NEH heuristic for initial construction
    2. Exhaustive best-insertion local search
    3. Swap neighborhood search
    4. Iterative improvement with time limit
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Stage 1: NEH initial solution
    seq = _neh_initial(processing_times)
    best_ms = _makespan_fast(processing_times, seq)
    
    # Stage 2: Local search with insertion and swap moves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Best insertion move
        new_seq, new_ms = _best_insertion(processing_times, seq)
        if new_ms < best_ms - 1e-12:
            seq = new_seq
            best_ms = new_ms
            improved = True
        
        if time.time() >= deadline:
            break
        
        # Best swap move
        new_seq, new_ms = _swap_improvement(processing_times, seq)
        if new_ms < best_ms - 1e-12:
            seq = new_seq
            best_ms = new_ms
            improved = True
    
    return list(seq)
# EVOLVE-BLOCK-END
