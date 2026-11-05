"""Rewritten program for flow-shop permutation task."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(pt, perm):
    """Compute makespan for a given permutation using DP."""
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    prev = [0.0] * n_machines
    for j in perm:
        curr = [0.0] * n_machines
        curr[0] = prev[0] + pt[j, 0]
        for m in range(1, n_machines):
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + pt[j, m]
        prev = curr
    return prev[-1]

def _neh_initial(pt):
    """NEH heuristic initial solution."""
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
    """Palmer heuristic (WSPT) initial solution."""
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
    """LPT (Longest Processing Time) on first machine."""
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

def _spt_initial(pt):
    """SPT (Shortest Processing Time) on first machine."""
    n_jobs, n_machines = pt.shape
    order = list(np.argsort(pt[:, 0]))
    
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

def _local_search(pt, perm, deadline):
    """
    Hill climbing with insertion neighborhood.
    Tries to move each job to every other position.
    """
    n_jobs, n_machines = pt.shape
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            job = current_perm[i]
            for k in range(n_jobs):
                if k == i or k == i - 1:
                    continue
                new_perm = current_perm[:i] + current_perm[i+1:]
                if k > i:
                    k -= 1
                new_perm = new_perm[:k] + [job] + new_perm[k:]
                new_ms = _makespan(pt, new_perm)
                if new_ms < current_ms:
                    current_perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break
            if improved:
                break
                
    return current_perm, current_ms

def _simulated_annealing(pt, perm, deadline, T0=100.0, cooling=0.99):
    """
    Simulated annealing with insertion neighborhood.
    """
    n_jobs, n_machines = pt.shape
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    T = T0
    rng = np.random.RandomState(123)
    
    while time.time() < deadline and T > 1e-5:
        i = rng.randint(n_jobs)
        k = rng.randint(n_jobs)
        if k == i or k == i - 1:
            continue
            
        job = current_perm[i]
        new_perm = current_perm[:i] + current_perm[i+1:]
        if k > i:
            k -= 1
        new_perm = new_perm[:k] + [job] + new_perm[k:]
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
    Construct a job permutation minimizing makespan using multi-start local search.
    """
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))
    
    # Generate multiple initial solutions
    candidates = []
    candidates.append(_neh_initial(pt))
    candidates.append(_palmer_initial(pt))
    candidates.append(_lpt_initial(pt))
    candidates.append(_spt_initial(pt))
    
    for seed in range(5):
        candidates.append(_random_initial(n_jobs, seed))
    
    best_perm = candidates[0]
    best_ms = _makespan(pt, best_perm)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
            
        perm, ms = _local_search(pt, cand, deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    if time.time() < deadline - 5.0:
        sa_deadline = deadline - 2.0
        perm, ms = _simulated_annealing(pt, best_perm, sa_deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    return best_perm

# EVOLVE-BLOCK-END
