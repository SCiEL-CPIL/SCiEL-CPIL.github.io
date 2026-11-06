"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, processing_times):
    """Compute makespan for a given job order."""
    n_machines = processing_times.shape[1]
    if n_machines == 0 or len(order) == 0:
        return 0.0
    
    # Use lists for faster access in inner loop
    completion = [0.0] * n_machines
    for job in order:
        p = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            c = completion[m]
            if c > prev:
                prev = c
            prev += p[m]
            completion[m] = prev
    return completion[-1]


def _compute_completion_matrix(order, processing_times):
    """
    Compute the full completion time matrix C[j, m] for the given order.
    """
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs == 0 or n_machines == 0:
        return np.zeros((0, n_machines))
    
    C = np.zeros((n_jobs, n_machines))
    for j, job in enumerate(order):
        p = processing_times[job]
        if j == 0:
            C[j, 0] = p[0]
        else:
            C[j, 0] = C[j-1, 0] + p[0]
        
        for m in range(1, n_machines):
            up = C[j-1, m]
            left = C[j, m-1]
            C[j, m] = (up if up > left else left) + p[m]
    return C


def _eval_insertion(order, i, j, processing_times):
    """
    Evaluate the makespan of inserting job at index i to position j.
    Returns the new makespan.
    Assumes j != i and j != i+1.
    """
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    # Create new order
    # Instead of slicing (which creates copies), we can simulate the sequence
    # But for simplicity and safety, let's just create the list.
    # Optimization: only recompute the affected part?
    # The dependency is global, so we must recompute from min(i, j) onwards?
    # Actually, if we remove i and insert at j:
    # If j < i: jobs j..i-1 shift right. Jobs i+1..end unchanged relative to each other but shifted in time?
    # No, the start times of jobs after the insertion point change.
    # So we must recompute from min(i, j) to end.
    
    # Let's just do a full recompute but with a fast loop.
    # We can avoid creating the full new_order list if we are clever, but let's stick to correctness first.
    
    new_order = order[:i] + order[i+1:]
    # Insert at j
    # Note: if j > i, the index in new_order is j-1 (since we removed i which was before j)
    # if j < i, the index in new_order is j
    if j > i:
        insert_pos = j - 1
    else:
        insert_pos = j
        
    new_order = new_order[:insert_pos] + [order[i]] + new_order[insert_pos:]
    
    return _makespan(new_order, processing_times)


def _neh_initialization(processing_times, seed=None):
    """
    NEH heuristic with optional randomization.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    totals = processing_times.sum(axis=1)
    perturbation = rng.rand(n_jobs) * 0.1 * (totals.max() - totals.min() + 1e-9)
    perturbed_totals = totals + perturbation
    job_order = list(np.argsort(-perturbed_totals))
    
    seq = [job_order[0]]
    
    for job in job_order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    
    return seq


def _fast_local_search(order, processing_times, time_limit=50.0):
    """
    Fast local search using insertion moves.
    """
    start_time = time.time()
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan(order, processing_times)
    
    best_order = list(order)
    best_ms = _makespan(best_order, processing_times)
    
    # We will use a simple first-improvement or best-improvement strategy.
    # Given the overhead of Python, best-improvement over all N^2 moves is expensive.
    # Let's try a hybrid: iterate through moves, keep track of best move in current pass.
    # If best move improves, apply it and restart pass? Or just apply and continue?
    # Standard Or-opt (insertion) local search.
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        best_move = None
        best_new_ms = best_ms
        
        # Iterate over all possible moves (i, j)
        for i in range(n_jobs):
            if time.time() - start_time >= time_limit:
                break
            for j in range(n_jobs):
                if j == i or j == i + 1:
                    continue
                
                ms = _eval_insertion(best_order, i, j, processing_times)
                if ms < best_new_ms:
                    best_new_ms = ms
                    best_move = (i, j)
        
        if best_move is not None:
            i, j = best_move
            # Apply move
            job = best_order[i]
            new_order = best_order[:i] + best_order[i+1:]
            if j > i:
                insert_pos = j - 1
            else:
                insert_pos = j
            best_order = new_order[:insert_pos] + [job] + new_order[insert_pos:]
            best_ms = best_new_ms
            improved = True
    
    return best_order, best_ms


def order_jobs(processing_times):
    """
    Main function: construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    best_order = None
    best_ms = float('inf')
    
    # Strategy: Multiple NEH restarts
    n_restarts = 10 if n_jobs < 30 else 5
    
    for r in range(n_restarts):
        if time.time() >= deadline - 10.0:
            break
        
        seed = r * 1000 + 42
        initial = _neh_initialization(processing_times, seed=seed)
        
        remaining = deadline - time.time() - 5.0
        if remaining > 5.0:
            improved, ms = _fast_local_search(initial, processing_times, time_limit=remaining)
            if ms < best_ms:
                best_ms = ms
                best_order = improved
    
    if best_order is None:
        best_order = _neh_initialization(processing_times)
    
    return best_order
# EVOLVE-BLOCK-END
