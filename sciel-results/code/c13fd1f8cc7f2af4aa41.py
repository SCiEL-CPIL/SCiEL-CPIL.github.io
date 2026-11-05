"""
Flow-shop permutation optimization using VND (Variable Neighborhood Descent)
and Simulated Annealing with Or-Opt neighborhood.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

TIME_LIMIT = 110.0  # Safety margin from 120s

def _makespan(pt, perm):
    """
    Compute makespan for a given permutation. O(N*M).
    Optimized for speed.
    """
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    # Use a flat array for completion times to avoid list allocation overhead
    # C[j, m] is completion time of job j on machine m
    # We only need the previous row to compute the current row?
    # No, in flow shop, C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
    # So we can maintain a 1D array `prev` which represents C[j-1, :]
    # And a scalar `curr_m_minus_1` which represents C[j, m-1]
    
    n_m = n_machines
    pt_local = pt
    prev = np.zeros(n_m)
    
    for j in perm:
        curr_m_minus_1 = 0.0
        for m in range(n_m):
            # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
            # prev[m] holds C[j-1, m]
            # curr_m_minus_1 holds C[j, m-1]
            val = prev[m] if prev[m] > curr_m_minus_1 else curr_m_minus_1
            curr_m_minus_1 = val + pt_local[j, m]
            prev[m] = curr_m_minus_1
            
    return prev[-1]

def _neh_initial(pt):
    """
    NEH heuristic: Sort jobs by total processing time descending.
    Insert each job into the best position in the current sequence.
    """
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _palmer_initial(pt):
    """
    Palmer's heuristic.
    """
    n_jobs, n_machines = pt.shape
    weights = np.arange(n_machines, 0, -1)
    priorities = pt.dot(weights)
    order = list(np.argsort(-priorities))
    
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _lpt_initial(pt):
    """
    LPT: Sort by longest processing time on first machine.
    """
    n_jobs, n_machines = pt.shape
    order = list(np.argsort(-pt[:, 0]))
    
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _random_initial(n_jobs, seed):
    """Random initial solution."""
    rng = np.random.RandomState(seed)
    return rng.permutation(n_jobs).tolist()

def _tabu_search(pt, perm, deadline, tabu_length=10, max_block_size=3):
    """
    Tabu Search with block-adjacent insertion neighborhood.
    Moves a block of size 1, 2, or 3 to any other position.
    Uses a dynamic tabu list to prevent cycling.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm, _makespan(pt, perm)
        
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    # Tabu list stores (start_index, end_index, target_index) or similar
    # To simplify, we store the move that generated the current solution
    # A move is defined by: job/block to move, from position, to position
    # We store a set of (job_id, from_idx, to_idx) but indices change.
    # Better: Store the set of jobs that are tabu to move from their current position?
    # Or store the last L moves.
    # Let's store a list of (block_start, block_end, insert_pos) that are tabu.
    # Since indices shift, this is tricky.
    # Alternative: Store the jobs in the block that are tabu.
    # If a job is in the tabu list, we cannot move it (or move it back to where it came from).
    # Standard approach for insertion:
    # Move job i to position k.
    # Tabu list contains (i, k) for the last L moves?
    # Let's use a simpler mechanism:
    # Tabu list is a list of jobs that cannot be moved.
    # When we move a block, we add all jobs in the block to the tabu list.
    # We remove the oldest jobs if the list exceeds tabu_length.
    
    tabu_jobs = [] # List of job IDs
    tabu_len = tabu_length
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        best_move_found = False
        best_delta = float('inf')
        best_new_perm = None
        best_new_ms = float('inf')
        best_move_info = None
        
        # Generate neighbors: move a block of size 1, 2, or 3
        # To limit computation, we don't evaluate ALL neighbors if N is large.
        # We sample or limit the scope.
        # For N < 50, we can do all. For N > 50, we need to be careful.
        
        max_block = min(max_block_size, n_jobs)
        
        # We iterate over possible blocks and destinations
        # This is O(N^3 * BlockSize) which is too slow for large N.
        # We will restrict the search space or use a greedy approach within TS.
        
        # Strategy:
        # 1. Try all single job moves (N*(N-1) ~ N^2)
        # 2. If N is small, try block moves.
        
        if n_jobs <= 30:
            # Full neighborhood search for small N
            for block_size in range(1, max_block + 1):
                for i in range(n_jobs - block_size + 1):
                    block = current_perm[i:i+block_size]
                    # Check if any job in block is tabu
                    # If yes, skip unless it's an aspiration move (better than best)
                    is_tabu = False
                    for job in block:
                        if job in tabu_jobs:
                            is_tabu = True
                            break
                    
                    for k in range(n_jobs - block_size + 1):
                        if k == i or k == i - 1:
                            continue
                        
                        # Calculate new makespan
                        # To speed up, we can use a fast evaluation.
                        # For now, use _makespan.
                        
                        new_perm = current_perm[:i] + current_perm[i+block_size:]
                        new_perm = new_perm[:k] + block + new_perm[k:]
                        
                        new_ms = _makespan(pt, new_perm)
                        
                        # Aspiration criterion: accept if better than global best
                        if new_ms < best_ms:
                            # Accept even if tabu
                            if new_ms < best_new_ms:
                                best_new_ms = new_ms
                                best_new_perm = new_perm
                                best_move_info = (i, block_size, k)
                                best_delta = new_ms - current_ms
                                best_move_found = True
                        elif not is_tabu:
                            # Accept if not tabu and best among non-tabu
                            if new_ms < best_new_ms:
                                best_new_ms = new_ms
                                best_new_perm = new_perm
                                best_move_info = (i, block_size, k)
                                best_delta = new_ms - current_ms
                                best_move_found = True
        else:
            # Large N: Sample moves to keep it fast
            num_samples = 50
            for _ in range(num_samples):
                block_size = rng.randint(1, min(3, n_jobs))
                i = rng.randint(0, n_jobs - block_size)
                k = rng.randint(0, n_jobs - block_size)
                if k == i or k == i - 1:
                    continue
                
                block = current_perm[i:i+block_size]
                is_tabu = any(job in tabu_jobs for job in block)
                
                new_perm = current_perm[:i] + current_perm[i+block_size:]
                new_perm = new_perm[:k] + block + new_perm[k:]
                
                new_ms = _makespan(pt, new_perm)
                
                if new_ms < best_ms or not is_tabu:
                    if new_ms < best_new_ms:
                        best_new_ms = new_ms
                        best_new_perm = new_perm
                        best_move_info = (i, block_size, k)
                        best_delta = new_ms - current_ms
                        best_move_found = True
        
        if not best_move_found:
            # No valid move found (all tabu and not aspirational), or no improvement
            # If no move found at all, break
            break
            
        # Update current solution
        current_perm = best_new_perm
        current_ms = best_new_ms
        
        # Update best solution
        if current_ms < best_ms:
            best_ms = current_ms
            best_perm = current_perm[:]
            
        # Update Tabu List
        i, block_size, k = best_move_info
        moved_jobs = current_perm[k:k+block_size] # The block is now at k
        for job in moved_jobs:
            if job in tabu_jobs:
                tabu_jobs.remove(job)
            tabu_jobs.append(job)
            
        if len(tabu_jobs) > tabu_len:
            tabu_jobs.pop(0)
            
        # Dynamic tabu length: increase if no improvement for a while?
        # Simple version: keep fixed or slight increase.
        # Let's keep it fixed for simplicity and stability.
        
    return best_perm, best_ms

def _simulated_annealing(pt, perm, deadline, T0=None, cooling=0.995):
    """
    Simulated Annealing with Or-Opt neighborhood (size 1, 2, 3).
    Kept as a backup if Tabu Search is not sufficient, but we will prioritize Tabu.
    """
    n_jobs = len(perm)
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    if T0 is None:
        T0 = max(10.0, current_ms * 0.5)
        
    T = T0
    rng = np.random.RandomState(42)
    
    max_block_size = 3
    
    while time.time() < deadline and T > 1e-6:
        block_size = rng.randint(1, max_block_size + 1)
        if n_jobs < block_size:
            block_size = n_jobs
            
        i = rng.randint(0, n_jobs - block_size + 1)
        k = rng.randint(0, n_jobs - block_size + 1)
        
        if k == i or k == i - 1:
            continue
            
        block = current_perm[i:i+block_size]
        
        new_perm = current_perm[:i] + current_perm[i+block_size:]
        new_perm = new_perm[:k] + block + new_perm[k:]
        
        new_ms = _makespan(pt, new_perm)
        
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
                
        T *= cooling
        
    return best_perm, best_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using Tabu Search.
    """
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))
    
    # Generate initial candidates
    candidates = []
    
    # 1. NEH
    candidates.append(_neh_initial(pt))
    
    # 2. Palmer
    candidates.append(_palmer_initial(pt))
    
    # 3. LPT
    candidates.append(_lpt_initial(pt))
    
    # 4. Random starts
    for seed in range(4):
        candidates.append(_random_initial(n_jobs, seed))
        
    best_perm = candidates[0]
    best_ms = _makespan(pt, best_perm)
    
    # Phase 1: Tabu Search on each candidate
    # Allocate time: ~10s for TS per candidate
    ts_deadline_per_cand = 10.0
    
    for cand in candidates:
        if time.time() > deadline - 5.0:  # Leave 5s for safety
            break
        cand_deadline = time.time() + ts_deadline_per_cand
        if cand_deadline > deadline - 2.0:
            cand_deadline = deadline - 2.0
            
        perm, ms = _tabu_search(pt, cand, cand_deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    # Phase 2: If time remains, run Tabu Search again on the best result with longer time
    if time.time() < deadline - 5.0:
        ts_deadline = deadline - 2.0
        perm, ms = _tabu_search(pt, best_perm, ts_deadline, tabu_length=20, max_block_size=3)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    # Ensure we return a list of ints
    return [int(x) for x in best_perm]

# EVOLVE-BLOCK-END
