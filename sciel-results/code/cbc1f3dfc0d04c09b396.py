"""Rewritten program for flow-shop permutation task."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(pt, perm):
    """
    Compute makespan for a given permutation.
    Uses a simple DP approach which is O(N*M).
    """
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    # Use lists for speed, avoiding numpy overhead for small M
    # C[m] represents the completion time of the current job on machine m
    # We iterate through jobs.
    # prev[m] is completion time of previous job on machine m
    prev = [0.0] * n_machines
    
    # Local variable for speed
    pt_local = pt
    n_m = n_machines
    
    for j in perm:
        curr = [0.0] * n_m
        curr[0] = prev[0] + pt_local[j, 0]
        for m in range(1, n_m):
            # max(prev[m], curr[m-1]) + pt[j, m]
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
        # Try inserting job at each position
        for pos in range(len(current) + 1):
            # Optimization: We can compute the makespan of the insertion
            # without creating a new list every time if we were smarter,
            # but for N<100, creating the list is fast enough.
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _palmer_initial(pt):
    """
    Palmer's heuristic (WSPT variant).
    Priority index: sum_{m=1}^{M} (M - m + 1) * t_{jm}
    """
    n_jobs, n_machines = pt.shape
    weights = np.arange(n_machines, 0, -1)
    priorities = pt.dot(weights)
    order = list(np.argsort(-priorities))
    
    # Refine with insertion similar to NEH but starting from Palmer order
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

def _local_search_first_improvement(pt, perm, deadline):
    """
    Local search using the insertion neighborhood (move a job to any other position).
    Uses a first-improvement strategy: scan the permutation, if a move improves,
    take it and restart the scan from the beginning.
    """
    n_jobs = len(perm)
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
            # Try moving job from i to every other position k
            # k != i and k != i-1 (since moving to i-1 is effectively a swap with previous,
            # but in insertion neighborhood, we remove i and insert at k.
            # If k > i, the indices shift.
            for k in range(n_jobs):
                if k == i or k == i - 1:
                    continue
                
                # Construct new permutation
                # Remove job at i
                new_perm = current_perm[:i] + current_perm[i+1:]
                # Adjust k if k > i
                if k > i:
                    k -= 1
                # Insert job at k
                new_perm = new_perm[:k] + [job] + new_perm[k:]
                
                new_ms = _makespan(pt, new_perm)
                if new_ms < current_ms:
                    current_perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break # Restart outer loop
            if improved:
                break
                
    return current_perm, current_ms

def _simulated_annealing(pt, perm, deadline, T0=None, cooling=0.995):
    """
    Simulated Annealing with insertion neighborhood.
    """
    n_jobs = len(perm)
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    if T0 is None:
        # Set initial temperature based on problem scale
        T0 = max(10.0, current_ms * 0.5)
        
    T = T0
    rng = np.random.RandomState(42)
    
    # Estimate max iterations based on time
    # We don't know how fast each iteration is, so we rely on time checks.
    
    while time.time() < deadline and T > 1e-6:
        # Pick a random move: remove a job and insert elsewhere
        i = rng.randint(n_jobs)
        k = rng.randint(n_jobs)
        if k == i or k == i - 1:
            continue
            
        job = current_perm[i]
        
        # Construct new permutation
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
    # 150s limit, leave 5s margin
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
    
    # 3. Random starts
    for seed in range(5):
        candidates.append(_random_initial(n_jobs, seed))
        
    best_perm = candidates[0]
    best_ms = _makespan(pt, best_perm)
    
    # Phase 1: Local Search on each candidate
    # Allocate time evenly or use a deadline for each
    ls_deadline = deadline - 20.0 # Leave 20s for SA
    
    for cand in candidates:
        if time.time() > ls_deadline:
            break
        perm, ms = _local_search_first_improvement(pt, cand, ls_deadline)
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
            
    # Phase 3: If time remains, try SA from other good local optima?
    # For now, just return the best found.
    
    return best_perm

# EVOLVE-BLOCK-END
