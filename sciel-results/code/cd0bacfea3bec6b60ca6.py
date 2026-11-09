"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan_fast(perm, pt):
    """Faster makespan computation using list-based perm."""
    if len(perm) == 0:
        return 0.0
    m = pt.shape[1]
    c = [0.0] * m
    for job_idx in perm:
        prev_c = 0.0
        row = pt[job_idx]
        for k in range(m):
            val = row[k]
            if c[k] > prev_c:
                c[k] = c[k] + val
            else:
                c[k] = prev_c + val
            prev_c = c[k]
    return c[-1]

def _neh(processing_times):
    """NEH heuristic: build solution by inserting jobs in order of decreasing total processing time."""
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = []
    for job in order:
        best_pos = -1
        best_ms = float('inf')
        for k in range(len(perm) + 1):
            trial = perm[:k] + [job] + perm[k:]
            ms = _makespan_fast(trial, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = k
        perm.insert(best_pos, job)
    
    return perm

def _or_opt_local_search(perm, pt, block_sizes=(1, 2, 3)):
    """
    Local search using Or-opt moves.
    Tries to move blocks of size 1, 2, 3 to better positions.
    Uses First Improvement strategy within a pass, repeating until convergence.
    """
    n = len(perm)
    if n <= 1:
        return list(perm)
    
    current_perm = list(perm)
    current_ms = _makespan_fast(current_perm, pt)
    
    improved = True
    # Limit iterations to prevent infinite loops on very large instances or bugs
    max_passes = 100 
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        for k in block_sizes:
            if k > n:
                continue
            for i in range(n - k + 1):
                # Extract block
                block = current_perm[i:i+k]
                # Remove block
                without_block = current_perm[:i] + current_perm[i+k:]
                
                best_insert_pos = -1
                best_ms_trial = current_ms
                
                # Try inserting block at all valid positions
                # Positions range from 0 to len(without_block)
                for j in range(len(without_block) + 1):
                    # Skip if it results in the same permutation (original position)
                    # Original position was i.
                    # If we insert at j, the block starts at j in the new perm.
                    # If j == i, it's the same.
                    # Also, if we insert at i+1, it's effectively the same relative to the removed part?
                    # Actually, removing [i:i+k] and inserting at i puts it back.
                    # Inserting at i+1 means it goes after the element that was at i+k? No.
                    # Let's just check if the resulting perm is identical.
                    if j == i:
                        continue
                    
                    trial_perm = without_block[:j] + block + without_block[j:]
                    
                    # Optional optimization: quick check if this is the same as current
                    # But _makespan_fast is fast enough for small n.
                    ms = _makespan_fast(trial_perm, pt)
                    
                    if ms < best_ms_trial:
                        best_ms_trial = ms
                        best_insert_pos = j
                        
                if best_insert_pos != -1:
                    # Apply the best move found for this block size and start index
                    without_block = current_perm[:i] + current_perm[i+k:]
                    current_perm = without_block[:best_insert_pos] + block + without_block[best_insert_pos:]
                    current_ms = best_ms_trial
                    improved = True
                    # Break out of i-loop? No, we can try other blocks.
                    # But usually, applying one move and restarting the scan is safer for "First Improvement" 
                    # to avoid cascading bad moves in one pass.
                    # However, "Best Improvement" per pass is often better.
                    # Let's stick to finding the single best move across all i and k in this pass?
                    # That would be O(n^3 * m). 
                    # Current logic: For each i, find best j. If found, apply and set improved=True.
                    # This is "First Best per i". It might miss a better global move in the same pass.
                    # To be more aggressive, we could store the global best move for the pass.
                    # Given time constraints, this "greedy per i" is a good balance.
                    
    return current_perm

def _perturb(perm, pt, rng, strength=3):
    """
    Perturb the permutation by applying 'strength' random Or-opt moves.
    """
    n = len(perm)
    if n <= 1:
        return list(perm)
    
    current_perm = list(perm)
    
    for _ in range(strength):
        # Random block size 1, 2, or 3
        k = rng.randint(1, min(3, n))
        # Random start index
        i = rng.randint(0, n - k)
        
        # Extract and remove
        block = current_perm[i:i+k]
        without_block = current_perm[:i] + current_perm[i+k:]
        
        # Random insertion position
        j = rng.randint(0, len(without_block))
        
        # Insert
        current_perm = without_block[:j] + block + without_block[j:]
        
    return current_perm

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin for 120s limit
    
    n, m = processing_times.shape
    
    if n <= 1:
        return list(range(n))
    
    # 1. Initial Solution: NEH
    best_perm = _neh(processing_times)
    best_ms = _makespan_fast(best_perm, processing_times)
    
    # 2. Initial Local Search
    best_perm = _or_opt_local_search(best_perm, processing_times)
    best_ms = _makespan_fast(best_perm, processing_times)
    
    rng = np.random.RandomState(42)
    
    current_perm = list(best_perm)
    current_ms = best_ms
    
    # 3. Iterated Local Search (ILS)
    # We use a simple acceptance criterion: accept if better, or with small probability if worse (SA-like)
    # But strict greedy ILS (accept only if better) is often robust.
    # Let's use a hybrid: Accept if better, or if worse but within a small delta and time is early?
    # Standard ILS accepts only improving moves to stay in the basin, relying on perturbation to escape.
    
    while time.time() < deadline - 5.0:
        # Perturb
        # Increase strength slightly if stuck? Keep simple for now.
        perturbed_perm = _perturb(current_perm, processing_times, rng, strength=2)
        
        # Local Search
        if time.time() > deadline - 5.0:
            break
            
        new_perm = _or_opt_local_search(perturbed_perm, processing_times)
        new_ms = _makespan_fast(new_perm, processing_times)
        
        # Acceptance
        if new_ms < current_ms:
            # Better than current, accept
            current_perm = new_perm
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
        else:
            # Worse or equal. 
            # To escape local minima, we might accept worse solutions occasionally.
            # However, pure greedy ILS with good perturbation is often sufficient.
            # Let's try a simple SA-like acceptance to diversify.
            # T is not explicitly managed, so we use a fixed small probability for worse moves?
            # Or just reject and perturb again from current?
            # Re-perturbing from the same current solution is standard ILS.
            pass
            
    return list(best_perm)
# EVOLVE-BLOCK-END
