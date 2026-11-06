"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation using the recurrence."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    # C[j][m] = max(C[j-1][m], C[j][m-1]) + P[perm[j]][m]
    # We only need the last row's final value (makespan)
    # But we need the full row for the recurrence
    prev_row = np.zeros(n_machines)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        cur_row = np.empty(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        for m in range(1, n_machines):
            cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
    return prev_row[-1]


def _neh_order(processing_times):
    """NEH heuristic: insert jobs one by one in order of decreasing total processing time."""
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    # Sort jobs by total processing time descending
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Build permutation by inserting each job in the best position
    perm = []
    for job in order:
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


def _two_opt_improve(perm, processing_times, max_iter=5000):
    """Improve permutation using 2-opt (swap) neighborhood search."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    current_ms = _makespan(perm, processing_times)
    best_perm = list(perm)
    best_ms = current_ms
    
    improved = True
    iterations = 0
    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        for i in range(n_jobs - 1):
            for j in range(i + 1, n_jobs):
                # Swap i and j
                trial = list(perm)
                trial[i], trial[j] = trial[j], trial[i]
                ms = _makespan(trial, processing_times)
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    best_perm = trial
                    perm = trial
                    improved = True
                    break
            if improved:
                break
    
    return best_perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses NEH initialization followed by 2-opt local search with deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Initial solution via NEH
    best_perm = _neh_order(processing_times)
    best_ms = _makespan(best_perm, processing_times)
    
    # Check time
    if time.time() > deadline - 10.0:
        return best_perm
    
    # 2-opt improvement on the NEH solution
    improved_perm = _two_opt_improve(best_perm, processing_times, max_iter=10000)
    improved_ms = _makespan(improved_perm, processing_times)
    
    if improved_ms < best_ms:
        best_perm = improved_perm
        best_ms = improved_ms
    
    # Check if we have time for random restarts
    elapsed = time.time() - start_time
    if elapsed < deadline - start_time - 15.0 and n_jobs <= 50:
        # Try a few random restarts with local search
        np.random.seed(42)
        max_restarts = 5
        for r in range(max_restarts):
            if time.time() > deadline - 10.0:
                break
            # Random permutation
            rand_perm = list(np.random.permutation(n_jobs))
            # Quick local search on random start
            local_perm = _two_opt_improve(rand_perm, processing_times, max_iter=2000)
            local_ms = _makespan(local_perm, processing_times)
            if local_ms < best_ms - 1e-9:
                best_ms = local_ms
                best_perm = local_perm
    
    # Final polish: one more 2-opt pass with more iterations if time permits
    if time.time() < deadline - 5.0:
        final_perm = _two_opt_improve(best_perm, processing_times, max_iter=5000)
        final_ms = _makespan(final_perm, processing_times)
        if final_ms < best_ms:
            best_perm = final_perm
    
    return best_perm
# EVOLVE-BLOCK-END
