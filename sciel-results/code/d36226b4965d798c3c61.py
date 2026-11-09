# EVOLVE-BLOCK-START
import time
import numpy as np

def _makespan(perm, pt):
    """Compute makespan for a permutation. O(n*m)."""
    n = len(perm)
    m = pt.shape[1]
    if m == 0 or n == 0:
        return 0.0
    
    # prev_completion[i] is the completion time of job perm[i] on the previous machine
    # For machine 0, it is the cumulative sum of processing times on machine 0
    prev_completion = np.cumsum(pt[perm, 0])
    
    for k in range(1, m):
        curr_completion = np.empty(n, dtype=np.float64)
        curr_completion[0] = prev_completion[0] + pt[perm[0], k]
        # Vectorized update is hard due to dependency on curr_completion[i-1]
        # But we can loop. For typical n, this is fast.
        for i in range(1, n):
            # curr_completion[i] = max(prev_completion[i], curr_completion[i-1]) + pt[perm[i], k]
            # Using max is slightly slower than if/else in Python loops, but NumPy array access is the bottleneck.
            # Let's stick to simple loop for clarity and moderate speed.
            v1 = prev_completion[i]
            v2 = curr_completion[i-1]
            curr_completion[i] = (v1 if v1 > v2 else v2) + pt[perm[i], k]
        prev_completion = curr_completion
    
    return float(prev_completion[-1])

def _neh_heuristic(pt, sort_key='sum'):
    """NEH heuristic for flow shop. Returns permutation."""
    n, m = pt.shape
    
    if sort_key == 'sum':
        totals = pt.sum(axis=1)
    elif sort_key == 'mean':
        totals = pt.mean(axis=1)
    elif sort_key == 'min':
        totals = pt.min(axis=1)
    elif sort_key == 'max':
        totals = pt.max(axis=1)
    else:
        totals = pt.sum(axis=1)
    
    order = np.argsort(-totals)
    
    perm = []
    
    for job in order:
        if not perm:
            perm = [job]
            continue
        
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(np.array(new_perm, dtype=np.int64), pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm.insert(best_pos, job)
    
    return np.array(perm, dtype=np.int64)

def _local_search_insertion(perm, pt, deadline=None):
    """
    Greedy Insertion Local Search.
    For each job, try moving it to every other position.
    Accept the best move if it improves makespan.
    Repeat until no improvement is found.
    """
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, pt)
    
    improved = True
    while improved:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                return best_perm
            
            job = best_perm[i]
            # Remove job at i
            # We create a list without job i
            remaining = np.delete(best_perm, i)
            
            best_pos_for_job = -1
            best_ms_for_job = float('inf')
            
            # Try inserting job at every position in remaining
            for pos in range(len(remaining) + 1):
                # If pos > i, the effective position in the original array shifts
                # But we are just constructing a new permutation, so we just insert into 'remaining'
                new_perm = np.insert(remaining, pos, job)
                ms = _makespan(new_perm, pt)
                
                if ms < best_ms_for_job:
                    best_ms_for_job = ms
                    best_pos_for_job = pos
            
            # Check if the best position is different from original position
            # Original position i corresponds to position i in the new array IF we didn't remove it?
            # No, we removed it. 
            # If we insert back at the same logical place, ms should be same.
            # We only accept if strictly better.
            if best_ms_for_job < best_ms - 1e-9:
                best_perm = np.insert(remaining, best_pos_for_job, job)
                best_ms = best_ms_for_job
                improved = True
                
    return best_perm

def _simulated_annealing(perm, pt, deadline, initial_temp=None, cooling_rate=0.99):
    """
    Simulated Annealing using random insertion moves.
    """
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, pt)
    
    current_perm = perm.copy()
    current_ms = best_ms
    
    if initial_temp is None:
        initial_temp = max(1.0, current_ms * 0.1)
    
    T = initial_temp
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Random insertion move
        # Pick a random job index i
        i = rng.integers(0, n)
        # Pick a random target position j
        # To avoid trivial no-ops, we can just pick j randomly from 0 to n
        # Move job at i to position j
        job = current_perm[i]
        remaining = np.delete(current_perm, i)
        
        # If i < j, inserting at j in the new array (size n-1) might need adjustment?
        # No, np.insert handles indices in the new array.
        # However, to ensure we actually move it, we should probably ensure the target is different.
        # But random insertion is fine.
        
        # Choose target pos in [0, n-1] (size of remaining is n-1, so insert pos 0..n-1)
        j = rng.integers(0, n)
        
        new_perm = np.insert(remaining, j, job)
        new_ms = _makespan(new_perm, pt)
        
        delta = new_ms - current_ms
        
        if delta < 0:
            # Accept improvement
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm.copy()
        else:
            # Accept worse solution with probability exp(-delta/T)
            if T > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_perm = new_perm
                    current_ms = new_ms
            
        T *= cooling_rate
        if T < 1e-6:
            T = 1e-6 # Floor temperature
    
    return best_perm

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    # Safety margin: 145s out of 150s
    deadline = start_time + 145.0
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    best_perm = None
    best_ms = float('inf')
    
    # 1. Generate multiple NEH variants
    keys = ['sum', 'mean', 'min', 'max']
    candidates = []
    
    for key in keys:
        if time.time() > deadline:
            break
        perm = _neh_heuristic(pt, key)
        ms = _makespan(perm, pt)
        candidates.append((perm, ms))
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # 2. Apply Insertion Local Search to each candidate
    if time.time() < deadline:
        for i, (perm, ms) in enumerate(candidates):
            if time.time() > deadline:
                break
            # Only refine if this candidate is promising or if it's the first one
            # To save time, we can limit LS iterations or just run it.
            # Insertion LS is fast.
            ls_perm = _local_search_insertion(perm, pt, deadline=deadline)
            ls_ms = _makespan(ls_perm, pt)
            
            if ls_ms < best_ms:
                best_ms = ls_ms
                best_perm = ls_perm
    
    # 3. Simulated Annealing from the best solution
    if time.time() < deadline:
        sa_perm = _simulated_annealing(best_perm, pt, deadline)
        sa_ms = _makespan(sa_perm, pt)
        if sa_ms < best_ms:
            best_ms = sa_ms
            best_perm = sa_perm

    return list(best_perm)
# EVOLVE-BLOCK-END
