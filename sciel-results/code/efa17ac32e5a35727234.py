"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

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
        return list(range(n)), 0.0
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
    
    return perm, _makespan_fast(perm, processing_times)

def _insertion_local_search(perm, pt, max_iters=500):
    """
    Insertion-based local search: try moving each job to every other position.
    Returns improved permutation and makespan.
    """
    n = len(perm)
    if n <= 1:
        return list(perm), _makespan_fast(perm, pt)
    
    current_perm = list(perm)
    current_ms = _makespan_fast(current_perm, pt)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n):
            job = current_perm[i]
            # Remove job from position i
            without_job = current_perm[:i] + current_perm[i+1:]
            # Try inserting at each position
            for k in range(n - 1):
                if k == i:
                    continue
                trial = without_job[:k] + [job] + without_job[k:]
                ms = _makespan_fast(trial, pt)
                if ms < current_ms:
                    current_perm = trial
                    current_ms = ms
                    improved = True
                    break
            if improved:
                break
    
    return current_perm, current_ms

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin for 120s limit
    
    n, m = processing_times.shape
    
    if n <= 1:
        return list(range(n))
    
    # 1. Initial Solution: NEH with insertion
    best_perm, best_ms = _neh(processing_times)
    
    # Apply insertion local search to the NEH solution
    best_perm, best_ms = _insertion_local_search(best_perm, processing_times, max_iters=100)
    
    # 2. Multi-restart Simulated Annealing with perturbation from best
    rng = np.random.RandomState(42)
    
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb the best solution to create a diverse starting point
        current_perm = list(best_perm)
        num_swaps = max(1, n // 10)
        for _ in range(num_swaps):
            i, j = rng.choice(n, 2, replace=False)
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        
        current_ms = _makespan_fast(current_perm, processing_times)
        
        # Simulated Annealing parameters tuned for the problem
        T = best_ms * 0.05
        T_min = best_ms * 0.001
        alpha = 0.9995
        
        # Inner loop: do SA steps until time runs out or T drops too low
        while time.time() < deadline - 10.0:
            if T < T_min:
                break
            
            # Generate neighbor by swap
            i, j = rng.choice(n, 2, replace=False)
            new_perm = current_perm[:]
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            
            new_ms = _makespan_fast(new_perm, processing_times)
            delta = new_ms - current_ms
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_perm = new_perm
                current_ms = new_ms
                
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_perm = current_perm[:]
            
            T *= alpha
        
        # Apply insertion local search to the SA result if it's reasonably close
        if current_ms <= best_ms * 1.05:
            polished_perm, polished_ms = _insertion_local_search(current_perm, processing_times, max_iters=50)
            if polished_ms < best_ms:
                best_ms = polished_ms
                best_perm = polished_perm
    
    # Final polish with more iterations if time allows
    if time.time() < deadline - 2.0:
        polished_perm, polished_ms = _insertion_local_search(best_perm, processing_times, max_iters=100)
        if polished_ms < best_ms:
            best_ms = polished_ms
            best_perm = polished_perm
    
    return list(best_perm)
# EVOLVE-BLOCK-END
