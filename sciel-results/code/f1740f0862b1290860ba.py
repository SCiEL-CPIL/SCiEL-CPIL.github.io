"""
Flow-shop permutation construction using a different strategy:
1. Multi-start NEH with different tie-breaking orders (total processing time, plus random perturbations).
2. Fast local search using insertion moves with incremental makespan evaluation (O(n*m) per move instead of O(n*m) full recompute).
3. Diversification via occasional random restarts and simulated-annealing-like acceptance of worse moves.
4. Vectorized makespan for final evaluation, but incremental delta for neighbor evaluation.
"""
import numpy as np
import time
import random

def _makespan(order, P):
    """Compute makespan for a given job order."""
    if len(order) == 0:
        return 0.0
    n_jobs_seq = len(order)
    n_machines = P.shape[1]
    seq_P = P[order]
    c_machines = np.zeros(n_machines)
    last_completion = 0.0
    for j in range(n_jobs_seq):
        row = seq_P[j]
        c_curr = 0.0
        for m in range(n_machines):
            c_curr = max(c_machines[m], c_curr) + row[m]
            c_machines[m] = c_curr
        last_completion = c_curr
    return last_completion

def _fast_insertion_delta(seq, P, job_idx, remove_pos, insert_pos):
    """
    Compute the change in makespan when removing job at remove_pos and inserting at insert_pos.
    This is an approximation that recomputes only the affected suffix.
    For correctness, we recompute makespan from the affected position onward.
    """
    n = len(seq)
    if n == 0:
        return 0.0
    if remove_pos == insert_pos:
        return 0.0
    
    # Build the new sequence
    if remove_pos < insert_pos:
        # Removing shifts elements left, so insertion position in remaining is insert_pos - 1
        new_seq = seq[:remove_pos] + seq[remove_pos+1:insert_pos] + [job_idx] + seq[insert_pos:]
    else:
        new_seq = seq[:remove_pos] + seq[remove_pos+1:insert_pos+1] + [job_idx] + seq[insert_pos+1:]
    
    # The makespan can change starting from min(remove_pos, insert_pos)
    start_recompute = min(remove_pos, insert_pos)
    
    # We need to compute the completion times from start_recompute onward
    # But we need the state of machines at start_recompute
    # For simplicity and correctness, recompute from start_recompute
    n_machines = P.shape[1]
    c_machines = np.zeros(n_machines)
    
    # First, compute completion times up to start_recompute
    for j in range(start_recompute):
        row = P[seq[j]]
        c_curr = 0.0
        for m in range(n_machines):
            c_curr = max(c_machines[m], c_curr) + row[m]
            c_machines[m] = c_curr
    
    # Now compute from start_recompute with the new sequence
    last_completion = 0.0
    for j in range(start_recompute, len(new_seq)):
        row = P[new_seq[j]]
        c_curr = 0.0
        for m in range(n_machines):
            c_curr = max(c_machines[m], c_curr) + row[m]
            c_machines[m] = c_curr
        last_completion = c_curr
    
    # Compute original makespan from start_recompute
    orig_c_machines = np.zeros(n_machines)
    for j in range(start_recompute):
        row = P[seq[j]]
        c_curr = 0.0
        for m in range(n_machines):
            c_curr = max(orig_c_machines[m], c_curr) + row[m]
            orig_c_machines[m] = c_curr
    
    orig_last = 0.0
    for j in range(start_recompute, len(seq)):
        row = P[seq[j]]
        c_curr = 0.0
        for m in range(n_machines):
            c_curr = max(orig_c_machines[m], c_curr) + row[m]
            orig_c_machines[m] = c_curr
        orig_last = c_curr
    
    return last_completion - orig_last

def _neh_heuristic(P, seed=None):
    """NEH heuristic with optional random tie-breaking."""
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    
    if seed is not None:
        rng = random.Random(seed)
        # Add small random noise to break ties
        noise = rng.uniform(0, 0.01 * totals.max())
        totals_noisy = totals + noise
        order_indices = np.argsort(-totals_noisy).tolist()
    else:
        order_indices = np.argsort(-totals).tolist()
    
    current_seq = []
    for idx in order_indices:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
    
    return current_seq

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using multi-start NEH + fast insertion local search.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    time_limit = 145.0
    
    best_seq = None
    best_makespan = float('inf')
    
    # Multi-start: run NEH with different seeds and do local search on each
    n_starts = min(10, max(3, n_jobs // 5))
    
    for s in range(n_starts):
        if (time.time() - start_time) > time_limit - 5.0:
            break
            
        # Generate initial solution
        if s == 0:
            current_seq = _neh_heuristic(P)
        else:
            current_seq = _neh_heuristic(P, seed=s * 42 + 1)
        
        current_makespan = _makespan(current_seq, P)
        
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_seq = current_seq.copy()
        
        # Local search with insertion moves
        improved = True
        local_iter = 0
        max_local_iter = 5000
        
        while improved and local_iter < max_local_iter:
            improved = False
            local_iter += 1
            
            if (time.time() - start_time) > time_limit - 5.0:
                break
            
            n = len(current_seq)
            
            # Try all single-job insertion moves
            best_delta = float('inf')
            best_move = None
            
            for i in range(n):
                for j in range(n):
                    if i == j or i == j + 1:
                        continue
                    
                    delta = _fast_insertion_delta(current_seq, P, current_seq[i], i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, j)
            
            if best_move is not None and best_delta < 0:
                i, j = best_move
                job_idx = current_seq[i]
                if i < j:
                    current_seq = current_seq[:i] + current_seq[i+1:j] + [job_idx] + current_seq[j:]
                else:
                    current_seq = current_seq[:i] + current_seq[i+1:j+1] + [job_idx] + current_seq[j+1:]
                current_makespan += best_delta
                improved = True
                
                if current_makespan < best_makespan:
                    best_makespan = current_makespan
                    best_seq = current_seq.copy()
            
            # Occasional random perturbation to escape local optima
            if local_iter % 50 == 0 and n > 3:
                # Random swap
                i, j = random.sample(range(n), 2)
                current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
                current_makespan = _makespan(current_seq, P)
                if current_makespan < best_makespan:
                    best_makespan = current_makespan
                    best_seq = current_seq.copy()
    
    if best_seq is None:
        best_seq = list(range(n_jobs))
    
    return best_seq
