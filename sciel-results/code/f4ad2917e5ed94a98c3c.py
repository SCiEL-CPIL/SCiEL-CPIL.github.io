"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently using the recurrence."""
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    # c[j] holds C[i][j] for the current job i
    # We iterate jobs in order; for each job, we compute completion times across machines
    c = np.zeros(n_machines)
    for job_idx in order:
        p_row = processing_times[job_idx]
        # c[j] = max(c[j], c[j-1]) + p[j]  where c[-1] = 0
        # We need to be careful: c[j-1] here is the new c[j-1] (current job), 
        # and c[j] is the old c[j] (previous job's completion on machine j)
        for j in range(n_machines):
            if j == 0:
                c[j] = c[j] + p_row[j]
            else:
                c[j] = max(c[j], c[j-1]) + p_row[j]
    return c[-1]

def _insertion_makespan_delta(processing_times, order, job_idx, pos):
    """
    Compute the change in makespan if we insert job_idx at position pos in order.
    This is expensive, so we use a full recomputation for accuracy in the heuristic.
    Actually, for a constructive heuristic, we can just compute the makespan of the 
    new permutation. But that's O(n*m) per insertion, and O(n) insertions, so O(n^2*m).
    For n up to maybe 50-100, this is fine.
    """
    new_order = order[:pos] + [job_idx] + order[pos:]
    return _makespan(processing_times, new_order)

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # --- Phase 1: Constructive Heuristic (Insertion-based) ---
    # Start with an empty sequence
    remaining = list(range(n_jobs))
    current_order = []
    
    # Initial seed: use NEH rule to pick the first job
    totals = processing_times.sum(axis=1)
    first_job = np.argsort(-totals)[0]
    current_order.append(first_job)
    remaining.remove(first_job)
    
    # Iteratively insert the next job at the position that minimizes makespan
    # We process jobs in order of decreasing total processing time (NEH-like)
    # But instead of just appending, we insert optimally.
    # Actually, a better constructive approach:
    # 1. Sort all jobs by some priority (e.g., total processing time)
    # 2. Start with the highest priority job
    # 3. For each subsequent job, find the best insertion position in the current sequence
    
    # Let's sort remaining jobs by total processing time descending
    remaining_sorted = sorted(remaining, key=lambda j: -totals[j])
    
    for job in remaining_sorted:
        best_pos = 0
        best_ms = float('inf')
        # Try all possible insertion positions
        for pos in range(len(current_order) + 1):
            ms = _insertion_makespan_delta(processing_times, current_order, job, pos)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current_order.insert(best_pos, job)
        
        if time.time() > deadline:
            break
    
    best_order = list(current_order)
    best_ms = _makespan(processing_times, best_order)
    
    # --- Phase 2: Local Search (Insertion-based) ---
    # Use a simple hill-climbing with insertion moves
    # To avoid getting stuck, we can use a simple simulated annealing or just 
    # multiple restarts with random perturbations if time permits.
    
    # For now, let's do a simple iterative improvement:
    # Repeat until no improvement or time runs out
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all possible insertion moves: remove job i, insert at position j
        n = len(best_order)
        for i in range(n):
            if time.time() > deadline:
                break
            job = best_order[i]
            # Remove job
            temp_order = best_order[:i] + best_order[i+1:]
            # Try inserting at all positions
            for j in range(len(temp_order) + 1):
                if j == i:
                    continue  # Same position
                new_order = temp_order[:j] + [job] + temp_order[j:]
                ms = _makespan(processing_times, new_order)
                if ms < best_ms:
                    best_ms = ms
                    best_order = new_order
                    improved = True
                    # Restart the loop since the order changed
                    break
            if improved:
                break
    
    # --- Phase 3: Simulated Annealing (if time remains) ---
    if time.time() < deadline:
        current = list(best_order)
        current_ms = best_ms
        T = 1.0
        cooling = 0.999
        min_T = 1e-4
        
        while T > min_T and time.time() < deadline:
            # Random move: pick two positions and swap, or remove and insert
            # Let's use insertion move for larger neighborhood
            n = len(current)
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                T *= cooling
                continue
            
            job = current[i]
            temp = current[:i] + current[i+1:]
            # Adjust j if i < j
            if i < j:
                j -= 1
            new_order = temp[:j] + [job] + temp[j:]
            
            new_ms = _makespan(processing_times, new_order)
            delta = new_ms - current_ms
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current = new_order
                current_ms = new_ms
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_order = list(current)
            
            T *= cooling
    
    return best_order
# EVOLVE-BLOCK-END
