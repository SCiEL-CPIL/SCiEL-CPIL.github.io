"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _makespan(perm, p):
    """Compute makespan for a given permutation on processing times p."""
    # p shape: (n_jobs, n_machines)
    n_machines = p.shape[1]
    # completion time on each machine
    # We can compute this efficiently
    # For each job in order, track completion on each machine
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + p[j][m]
    # But we can do it in O(n_jobs * n_machines) with a running array
    
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    
    # comp[m] = completion time of the last job processed on machine m
    comp = np.zeros(n_machines)
    for j in perm:
        for m in range(n_machines):
            if m == 0:
                comp[m] += p[j, m]
            else:
                comp[m] = max(comp[m-1], comp[m]) + p[j, m]
    return comp[-1]

def _makespan_vectorized(perm, p):
    """Vectorized makespan computation."""
    n_machines = p.shape[1]
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    
    # Extract processing times in permutation order
    p_perm = p[perm]  # shape (n_jobs, n_machines)
    
    # Use cumulative sum with max constraint
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + p[j][m]
    # This is equivalent to: C[j][m] = C[j-1][m] + p[j][m] if C[j-1][m] >= C[j][m-1] - p[j][m]
    # Actually, let's just do the loop but with numpy for the inner part
    
    comp = np.zeros(n_machines)
    for j in range(n_jobs):
        pj = p_perm[j]
        for m in range(n_machines):
            if m == 0:
                comp[m] += pj[m]
            else:
                comp[m] = max(comp[m-1], comp[m]) + pj[m]
    return comp[-1]

def _cds_heuristic(p):
    """CDS (Campbell, Dudek, Smith) heuristic for initial solution."""
    n_jobs, n_machines = p.shape
    
    # Compute total processing times
    totals = p.sum(axis=1)
    
    # Sort jobs by total processing time (descending)
    order = np.argsort(-totals)
    
    # CDS: start with NEH-like ordering, then try insertions
    # Actually, let's use a proper CDS or NEH
    
    # NEH heuristic
    return _neh_order(p)

def _neh_order(p):
    """NEH (Nawaz-Enscore-Ham) heuristic for initial solution."""
    n_jobs, n_machines = p.shape
    
    # Sort jobs by total processing time descending
    totals = p.sum(axis=1)
    order = np.argsort(-totals)
    
    # Build solution by inserting one job at a time
    solution = []
    for job in order:
        # Try all possible insertion positions
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(solution) + 1):
            new_sol = solution[:pos] + [job] + solution[pos:]
            ms = _makespan(new_sol, p)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        solution = solution[:best_pos] + [job] + solution[best_pos:]
    
    return solution

def _two_opt_improve(perm, p, max_iter=100):
    """2-opt style local search: try all adjacent swaps."""
    n_jobs = len(perm)
    current_ms = _makespan(perm, p)
    improved = True
    iterations = 0
    
    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        for i in range(n_jobs - 1):
            # Swap jobs at positions i and i+1
            new_perm = list(perm)
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            new_ms = _makespan(new_perm, p)
            if new_ms < current_ms:
                perm = new_perm
                current_ms = new_ms
                improved = True
    
    return perm, current_ms

def _or_opt_moves(perm, p):
    """Or-opt moves: relocate a single job to a different position."""
    n_jobs = len(perm)
    current_ms = _makespan(perm, p)
    improved = True
    
    while improved:
        improved = False
        for i in range(n_jobs):
            for j in range(n_jobs):
                if i == j or i == j - 1:
                    continue
                # Move job from position i to position j
                new_perm = list(perm)
                job = new_perm.pop(i)
                # Adjust j if i < j
                if i < j:
                    j -= 1
                new_perm.insert(j, job)
                new_ms = _makespan(new_perm, p)
                if new_ms < current_ms:
                    perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break
            if improved:
                break
    
    return perm, current_ms

def _vnd_search(perm, p, deadline):
    """Variable Neighborhood Descent search."""
    n_jobs = len(perm)
    current_ms = _makespan(perm, p)
    best_perm = list(perm)
    best_ms = current_ms
    
    # Neighborhoods: adjacent swap, 2-opt reversal, or-opt relocation
    while time.time() < deadline:
        # Neighborhood 1: Adjacent swaps (first improvement)
        improved = False
        for i in range(n_jobs - 1):
            if time.time() > deadline:
                break
            new_perm = list(perm)
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            new_ms = _makespan(new_perm, p)
            if new_ms < current_ms:
                perm = new_perm
                current_ms = new_ms
                improved = True
                break
        if improved:
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = list(perm)
            continue
        
        # Neighborhood 2: Or-opt moves (first improvement)
        improved = False
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            for j in range(n_jobs):
                if i == j or i == j - 1:
                    continue
                new_perm = list(perm)
                job = new_perm.pop(i)
                if i < j:
                    j -= 1
                new_perm.insert(j, job)
                new_ms = _makespan(new_perm, p)
                if new_ms < current_ms:
                    perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break
            if improved:
                break
        if improved:
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = list(perm)
            continue
        
        # Neighborhood 3: 2-opt reversals (first improvement)
        improved = False
        for i in range(n_jobs - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_jobs):
                if time.time() > deadline:
                    break
                new_perm = list(perm)
                new_perm[i:j+1] = new_perm[i:j+1][::-1]
                new_ms = _makespan(new_perm, p)
                if new_ms < current_ms:
                    perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break
            if improved:
                break
        if improved:
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = list(perm)
            continue
        
        # No improvement in any neighborhood, break
        break
    
    return best_perm, best_ms

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    p = np.asarray(processing_times)
    n_jobs, n_machines = p.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Initial solution using NEH heuristic
    perm = _neh_order(p)
    current_ms = _makespan(perm, p)
    best_perm = list(perm)
    best_ms = current_ms
    
    # VND search
    best_perm, best_ms = _vnd_search(perm, p, deadline)
    
    return best_perm
# EVOLVE-BLOCK-END
