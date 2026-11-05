"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(pt, perm):
    """Compute makespan for a given permutation of job indices."""
    n_machines = pt.shape[1]
    # For each machine, compute the cumulative completion time
    # C[m] = max(C[m-1], C_prev_job[m-1]) + pt[job, m]
    # We can vectorize this somewhat but for clarity we do it in a loop
    # Actually, let's use the standard formula:
    # C_{i,m} = max(C_{i-1,m}, C_{i,m-1}) + p_{i,m}
    # where i is job index in permutation, m is machine.
    
    # Efficient computation:
    # Let cumsum_m be the cumulative sum of processing times on machine m up to position i.
    # C_{i,m} = max(C_{i,m-1}, cumsum_m[i])
    # This is not quite right because C_{i,m-1} depends on previous jobs on machine m-1.
    
    # Let's just do the standard O(n*m) computation which is fast enough.
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    
    # C[i][m] for i in range(n_jobs), m in range(n_machines)
    # C[i][0] = sum of pt[perm[0:i+1], 0]
    # C[i][m] = max(C[i-1][m], C[i][m-1]) + pt[perm[i], m]
    
    # We can compute this row by row
    prev_row = np.zeros(n_machines)
    for i in range(n_jobs):
        job = perm[i]
        if i == 0:
            # First job: C[0][m] = C[0][m-1] + pt[job, m]
            # This is just cumulative sum
            current = np.cumsum(pt[job])
        else:
            current = np.zeros(n_machines)
            current[0] = prev_row[0] + pt[job, 0]
            for m in range(1, n_machines):
                current[m] = max(prev_row[m], current[m-1]) + pt[job, m]
        prev_row = current
    
    return prev_row[-1]


def _makespan_fast(pt, perm):
    """Faster makespan computation using vectorized numpy operations."""
    n_machines = pt.shape[1]
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    
    # Extract the submatrix for the permutation
    sub = pt[perm]
    
    # C[i][m] = max(C[i-1][m], C[i][m-1]) + sub[i][m]
    # We need to compute this iteratively.
    
    # Use a 2D array for C
    C = np.zeros((n_jobs, n_machines))
    C[0, 0] = sub[0, 0]
    
    # First row
    for m in range(1, n_machines):
        C[0, m] = C[0, m-1] + sub[0, m]
    
    # First column
    for i in range(1, n_jobs):
        C[i, 0] = C[i-1, 0] + sub[i, 0]
    
    # Fill in the rest
    for i in range(1, n_jobs):
        for m in range(1, n_machines):
            C[i, m] = max(C[i-1, m], C[i, m-1]) + sub[i, m]
    
    return C[-1, -1]


def _delta_makespan(pt, perm, i, j):
    """Compute the change in makespan when swapping adjacent jobs at positions i and i+1."""
    # This is a heuristic approximation for speed
    # Actually, let's just recompute for simplicity and correctness
    n_machines = pt.shape[1]
    n_jobs = len(perm)
    
    # Swap
    perm2 = list(perm)
    perm2[i], perm2[i+1] = perm2[i+1], perm2[i]
    
    # Compute makespan for both
    ms1 = _makespan_fast(pt, perm)
    ms2 = _makespan_fast(pt, perm2)
    
    return ms2 - ms1


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using simulated annealing
    with adjacent swaps and 2-opt moves.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    pt = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = pt.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Initial solution: NEH (Nawaz-Enscore-Ham) heuristic
    # Sort jobs by decreasing total processing time, then insert each job
    # at the position that minimizes the partial makespan
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Build initial permutation using insertion
    perm = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(pt, new_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    
    # Convert to list of ints
    perm = [int(x) for x in perm]
    
    best_perm = perm[:]
    best_ms = _makespan_fast(pt, perm)
    
    # Simulated annealing
    T = best_ms * 0.1  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.9995
    
    # Number of iterations per temperature
    # We'll use a time-based approach
    iters_per_temp = max(10, min(1000, n_jobs * 10))
    
    current_perm = perm[:]
    current_ms = best_ms
    
    rng = np.random.RandomState(42)
    
    while T > T_min:
        # Check deadline
        if time.time() > deadline:
            break
        
        for _ in range(iters_per_temp):
            if time.time() > deadline:
                break
            
            # Generate a neighbor: swap two random positions
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if i == j:
                continue
            
            # Ensure i < j for consistency
            if i > j:
                i, j = j, i
            
            # Swap
            new_perm = current_perm[:]
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            
            new_ms = _makespan_fast(pt, new_perm)
            delta = new_ms - current_ms
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_perm = new_perm
                current_ms = new_ms
                
                # Update best
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_perm = current_perm[:]
        
        T *= cooling_rate
    
    return best_perm
# EVOLVE-BLOCK-END
