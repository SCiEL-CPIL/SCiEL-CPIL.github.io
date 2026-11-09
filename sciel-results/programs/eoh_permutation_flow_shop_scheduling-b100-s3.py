"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, pt):
    """Compute makespan for a given permutation using vectorized operations."""
    if len(perm) == 0:
        return 0.0
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0.0
    
    # Precompute completion times using the recurrence:
    # C[j, k] = max(C[j-1, k], C[j, k-1]) + p[j, k]
    # We maintain C_prev (completion times for previous job across machines)
    # and compute C_curr for current job.
    
    C_prev = np.zeros(m)
    for j in perm:
        C_curr = np.zeros(m)
        C_curr[0] = C_prev[0] + pt[j, 0]
        # For k >= 1: C_curr[k] = max(C_prev[k], C_curr[k-1]) + pt[j, k]
        # This has a sequential dependency on C_curr[k-1], so we can't fully vectorize
        # But we can use a loop with numpy scalars which is faster than Python floats
        for k in range(1, m):
            C_curr[k] = max(C_prev[k], C_curr[k-1]) + pt[j, k]
        C_prev = C_curr
        
    return float(C_prev[-1])

def _makespan_from_prefix(perm, pt, start_idx):
    """
    Compute makespan starting from a prefix.
    Assumes perm[0:start_idx] has already been processed and returns the completion vector
    after processing perm[start_idx:].
    For efficiency in local search, we precompute the completion state up to start_idx.
    """
    if start_idx >= len(perm):
        # Only need to compute from the prefix
        C_prev = np.zeros(pt.shape[1])
        for j in perm[:start_idx]:
            C_curr = np.zeros(pt.shape[1])
            C_curr[0] = C_prev[0] + pt[j, 0]
            for k in range(1, pt.shape[1]):
                C_curr[k] = max(C_prev[k], C_curr[k-1]) + pt[j, k]
            C_prev = C_curr
        return float(C_prev[-1]) if len(perm) > 0 else 0.0
    
    m = pt.shape[1]
    C_prev = np.zeros(m)
    for j in perm:
        C_curr = np.zeros(m)
        C_curr[0] = C_prev[0] + pt[j, 0]
        for k in range(1, m):
            C_curr[k] = max(C_prev[k], C_curr[k-1]) + pt[j, k]
        C_prev = C_curr
    return float(C_prev[-1])

def _neh_heuristic(pt):
    """
    NEH (Nawaz, Enscore, Ham) heuristic.
    1. Sort jobs by total processing time in descending order.
    2. Insert each job into the position that minimizes makespan.
    """
    n, m = pt.shape
    if n == 0:
        return []
    
    # Calculate total processing time for each job
    totals = pt.sum(axis=1)
    # Sort indices by total processing time descending
    order = np.argsort(-totals)
    
    current_perm = []
    for job_idx in order:
        job = int(job_idx)
        best_pos = 0
        best_mk = float('inf')
        
        # Try inserting job at each position in current_perm
        for pos in range(len(current_perm) + 1):
            candidate_perm = current_perm[:pos] + [job] + current_perm[pos:]
            mk = _makespan(candidate_perm, pt)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    return current_perm

def _local_search_insertion(perm, pt, time_limit=None):
    """
    Perform a local search using insertion moves.
    For each job, try moving it to every other position.
    Accept the best move if it improves the makespan.
    Repeat until no improvement is found or time limit reached.
    """
    n = len(perm)
    current_mk = _makespan(perm, pt)
    improved = True
    start = time.time()
    
    while improved:
        if time_limit is not None and time.time() - start > time_limit:
            break
        improved = False
        best_move = None
        best_mk = current_mk
        
        for i in range(n):
            job = perm[i]
            # Try inserting job at every position j != i
            # We can skip i and i+1 as they result in the same permutation or minor shifts
            for j in range(n + 1):
                if j == i or j == i + 1:
                    continue
                
                # Create candidate permutation
                candidate = perm[:i] + perm[i+1:]
                insert_pos = j
                if j > i:
                    insert_pos = j - 1
                candidate = candidate[:insert_pos] + [job] + candidate[insert_pos:]
                
                cand_mk = _makespan(candidate, pt)
                
                if cand_mk < best_mk:
                    best_mk = cand_mk
                    best_move = candidate
        
        if best_move is not None:
            perm = best_move
            current_mk = best_mk
            improved = True
            
    return perm, current_mk

def _simulated_annealing(perm, pt, best_mk, best_perm, deadline, n):
    """
    Simulated Annealing with insertion moves.
    Returns (best_perm, best_mk, current_perm, current_mk)
    """
    current_perm = perm[:]
    current_mk = _makespan(current_perm, pt)
    
    # Adaptive temperature schedule
    T = current_mk * 0.15
    cooling = 0.9995
    min_T = 1e-6
    
    # Track best
    local_best_perm = best_perm[:]
    local_best_mk = best_mk
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Generate neighbor: Insertion move
        i = random.randint(0, n - 1)
        j = random.randint(0, n)
        if j == i or j == i + 1:
            # Skip trivial moves but still cool
            T *= cooling
            continue
            
        job = current_perm[i]
        # Remove job
        candidate = current_perm[:i] + current_perm[i+1:]
        # Insert job at j
        insert_pos = j
        if j > i:
            insert_pos = j - 1
        candidate = candidate[:insert_pos] + [job] + candidate[insert_pos:]
        
        cand_mk = _makespan(candidate, pt)
        delta = cand_mk - current_mk
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_perm = candidate
            current_mk = cand_mk
            
            if current_mk < local_best_mk:
                local_best_mk = current_mk
                local_best_perm = current_perm[:]
                
        T *= cooling
        
    return local_best_perm, local_best_mk, current_perm, current_mk

def order_jobs(processing_times):
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
        
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # 1. Initial solution via NEH
    current_perm = _neh_heuristic(processing_times)
    current_mk = _makespan(current_perm, processing_times)
    best_perm = current_perm[:]
    best_mk = current_mk
    
    # 2. Local Search: Insertion with time limit
    ls_time_limit = min(10.0, (deadline - time.time()) * 0.3)
    current_perm, current_mk = _local_search_insertion(current_perm, processing_times, time_limit=ls_time_limit)
    if current_mk < best_mk:
        best_perm = current_perm[:]
        best_mk = current_mk
        
    # 3. Simulated Annealing with multiple restarts
    # Run SA for the remaining time, with restarts if needed
    remaining_time = deadline - time.time()
    
    if remaining_time > 5.0:
        # First SA run: use most of remaining time
        sa_time1 = remaining_time * 0.6
        deadline1 = start_time + (115.0 - sa_time1)
        best_perm, best_mk, current_perm, current_mk = _simulated_annealing(
            current_perm, processing_times, best_mk, best_perm, deadline1, n
        )
        
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            # Second SA run from best solution with fresh temperature
            sa_time2 = remaining_time * 0.8
            deadline2 = start_time + (115.0 - sa_time2)
            # Restart from best solution
            best_perm, best_mk, current_perm, current_mk = _simulated_annealing(
                best_perm, processing_times, best_mk, best_perm, deadline2, n
            )
            
            # Final local search polish
            remaining_time = deadline - time.time()
            if remaining_time > 2.0:
                ls_time_limit = min(3.0, remaining_time * 0.5)
                current_perm, current_mk = _local_search_insertion(
                    current_perm, processing_times, time_limit=ls_time_limit
                )
                if current_mk < best_mk:
                    best_perm = current_perm[:]
                    best_mk = current_mk
                    
    return list(best_perm)
# EVOLVE-BLOCK-END
