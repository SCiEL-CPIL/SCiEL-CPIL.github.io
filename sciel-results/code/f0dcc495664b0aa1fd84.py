"""
Flow-shop permutation optimization using VND (Variable Neighborhood Descent)
and Simulated Annealing with Or-Opt neighborhood.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(pt, perm):
    """
    Compute makespan for a given permutation. O(N*M).
    """
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    prev = [0.0] * n_machines
    n_m = n_machines
    pt_local = pt
    
    for j in perm:
        curr = [0.0] * n_m
        curr[0] = prev[0] + pt_local[j, 0]
        for m in range(1, n_m):
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + pt_local[j, m]
        prev = curr
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

def _vnd_local_search(pt, perm, deadline):
    """
    Variable Neighborhood Descent (VND) with nested neighborhoods:
    N1: Adjacent swaps (2-opt style, swap i and i+1)
    N2: Or-Opt block moves (size 1, 2, 3)
    N3: Reversals (reverse a segment)
    
    Cycles through neighborhoods, accepting first improvement in each.
    """
    n_jobs = len(perm)
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    
    max_block_size = 3
    max_reversal_len = 5
    
    while time.time() < deadline:
        improved = False
        
        # N1: Adjacent swaps
        for i in range(n_jobs - 1):
            if time.time() > deadline:
                break
            new_perm = current_perm[:]
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            new_ms = _makespan(pt, new_perm)
            if new_ms < current_ms:
                current_perm = new_perm
                current_ms = new_ms
                improved = True
                break  # Restart from N1
        
        if improved:
            continue
        
        # N2: Or-Opt block moves (size 1, 2, 3)
        for block_size in range(1, max_block_size + 1):
            if time.time() > deadline:
                break
            for i in range(n_jobs - block_size + 1):
                if time.time() > deadline:
                    break
                block = current_perm[i:i+block_size]
                
                for k in range(n_jobs - block_size + 1):
                    if time.time() > deadline:
                        break
                    if k == i or k == i - 1:
                        continue
                    
                    new_perm = current_perm[:i] + current_perm[i+block_size:]
                    new_perm = new_perm[:k] + block + new_perm[k:]
                    
                    new_ms = _makespan(pt, new_perm)
                    if new_ms < current_ms:
                        current_perm = new_perm
                        current_ms = new_ms
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # N3: Reversals of segments up to max_reversal_len
        for seg_len in range(2, max_reversal_len + 1):
            if time.time() > deadline:
                break
            for i in range(n_jobs - seg_len + 1):
                if time.time() > deadline:
                    break
                new_perm = current_perm[:]
                new_perm[i:i+seg_len] = new_perm[i:i+seg_len][::-1]
                new_ms = _makespan(pt, new_perm)
                if new_ms < current_ms:
                    current_perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # No improvement in any neighborhood, stop
            break
    
    return current_perm, current_ms

def _simulated_annealing(pt, perm, deadline, T0=None, cooling=0.995):
    """
    Simulated Annealing with Or-Opt neighborhood (size 1, 2, 3).
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
    Construct a job permutation minimizing makespan using VND and SA.
    """
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    start_time = time.time()
    deadline = start_time + 145
    
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
    
    # Phase 1: VND Local Search on each candidate
    # Allocate time: ~30s for VND across candidates, ~110s for SA
    ls_deadline_per_cand = 8.0
    
    for cand in candidates:
        if time.time() > deadline - 100.0:  # Leave 100s for SA
            break
        cand_deadline = time.time() + ls_deadline_per_cand
        if cand_deadline > deadline - 100.0:
            cand_deadline = deadline - 100.0
            
        perm, ms = _vnd_local_search(pt, cand, cand_deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    # Phase 2: Simulated Annealing from the best solution
    if time.time() < deadline - 2.0:
        sa_deadline = deadline - 1.0
        perm, ms = _simulated_annealing(pt, best_perm, sa_deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    # Phase 3: If time remains, run VND again on the SA result
    if time.time() < deadline - 5.0:
        vnd_deadline = deadline - 2.0
        perm, ms = _vnd_local_search(pt, best_perm, vnd_deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    return best_perm

# EVOLVE-BLOCK-END
