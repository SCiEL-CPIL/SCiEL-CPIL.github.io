"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job permutation.
    
    processing_times: shape (n_jobs, n_machines)
    order: list/array of job indices
    Returns: scalar makespan
    """
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    if n_machines == 0:
        return 0.0
    
    # Vectorized makespan computation
    # For each machine m, the completion time is cumulative max of previous machine completion + processing time
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + P[j][m]
    
    # Extract processing times in order
    pt_ordered = processing_times[order]  # shape (n_jobs, n_machines)
    
    # C[0, m] = sum of first job's processing times up to machine m
    # C[j, m] = max(C[j, m-1], C[j-1, m]) + P[j, m]
    
    # Use iterative approach but vectorized over jobs where possible
    # Actually for flow shop, we need sequential computation over machines
    # Let's use a different approach: compute completion times
    
    # C[j] is the completion time of job j on the last machine
    # We need the full matrix for intermediate steps
    
    # Efficient method: 
    # completion[j, m] = completion[j-1, m] + P[j, m] if completion[j-1, m] >= completion[j, m-1]
    #                  = completion[j, m-1] + P[j, m] otherwise
    
    # Let's just do it with a loop over machines, vectorized over jobs
    n = n_jobs
    k = n_machines
    
    # C[j, m] for j in 0..n-1, m in 0..k-1
    # C[0, m] = sum_{i=0}^{m} P[0, i]
    # C[j, 0] = sum_{i=0}^{j} P[i, 0]
    
    # Initialize
    C = np.zeros((n, k))
    
    # First job
    C[0, :] = np.cumsum(pt_ordered[0, :])
    
    # First machine
    C[:, 0] = np.cumsum(pt_ordered[:, 0])
    
    # Fill in the rest
    for m in range(1, k):
        for j in range(1, n):
            C[j, m] = max(C[j, m-1], C[j-1, m]) + pt_ordered[j, m]
    
    return C[-1, -1]


def _makespan_fast(processing_times, order):
    """Faster makespan computation using vectorized operations where possible."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs == 0 or n_machines == 0:
        return 0.0
    
    pt_ordered = processing_times[order]
    n = n_jobs
    k = n_machines
    
    # For small instances, the simple loop is fine
    # For larger instances, we can try to vectorize
    
    if k == 1:
        return pt_ordered.sum()
    
    # Use a rolling approach
    # C_prev_m[j] = completion time of job j on machine m-1
    # C_cur_m[j] = completion time of job j on machine m
    
    # Machine 0
    C_prev = np.cumsum(pt_ordered[:, 0])
    
    for m in range(1, k):
        # C_cur[j] = max(C_prev[j], C_cur[j-1]) + pt[j, m]
        # This is a sequential dependency, hard to fully vectorize
        # But we can process in chunks or use a different formulation
        
        # Actually, let's just do the loop - it's O(n*k) which is fine
        C_cur = np.empty(n)
        C_cur[0] = C_prev[0] + pt_ordered[0, m]
        for j in range(1, n):
            C_cur[j] = max(C_prev[j], C_cur[j-1]) + pt_ordered[j, m]
        C_prev = C_cur
    
    return C_prev[-1]


def _neh_initialization(processing_times):
    """NEH heuristic for initial solution."""
    n_jobs, n_machines = processing_times.shape
    
    # Sort jobs by total processing time descending
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    # Insert each job at the position that minimizes makespan
    perm = []
    for job in order:
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(processing_times, new_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    
    return perm


def _insertion_local_search(processing_times, perm, max_iter=None):
    """Insertion-based local search: remove each job and reinsert at best position."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    if max_iter is None:
        max_iter = 50
    
    best_perm = list(perm)
    best_ms = _makespan_fast(processing_times, best_perm)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(n_jobs):
            # Remove job at position i
            job = best_perm[i]
            remaining = best_perm[:i] + best_perm[i+1:]
            
            # Try inserting at each position
            best_pos = i
            best_ms_candidate = best_ms
            for pos in range(n_jobs - 1):
                new_perm = remaining[:pos] + [job] + remaining[pos:]
                ms = _makespan_fast(processing_times, new_perm)
                if ms < best_ms_candidate:
                    best_ms_candidate = ms
                    best_pos = pos
            
            if best_ms_candidate < best_ms:
                best_perm = remaining[:best_pos] + [job] + remaining[best_pos:]
                best_ms = best_ms_candidate
                improved = True
        
        if not improved:
            break
    
    return best_perm


def _adjacent_swap_local_search(processing_times, perm, max_iter=None):
    """Adjacent swap local search."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    if max_iter is None:
        max_iter = 50
    
    best_perm = list(perm)
    best_ms = _makespan_fast(processing_times, best_perm)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(n_jobs - 1):
            # Swap adjacent jobs i and i+1
            new_perm = best_perm[:]
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            ms = _makespan_fast(processing_times, new_perm)
            if ms < best_ms:
                best_perm = new_perm
                best_ms = ms
                improved = True
        
        if not improved:
            break
    
    return best_perm


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    processing_times = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Generate initial solutions using different heuristics
    best_perm = None
    best_ms = np.inf
    
    # NEH initialization
    if time.time() < deadline:
        perm_neh = _neh_initialization(processing_times)
        ms = _makespan_fast(processing_times, perm_neh)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm_neh
    
    # Simple LPT (Longest Processing Time) - the original seed
    if time.time() < deadline:
        perm_lpt = list(np.argsort(-processing_times.sum(axis=1)))
        ms = _makespan_fast(processing_times, perm_lpt)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm_lpt
    
    # V-shaped initialization
    if time.time() < deadline:
        totals = processing_times.sum(axis=1)
        sorted_indices = np.argsort(-totals)
        half = n_jobs // 2
        front = sorted_indices[:half]
        back = sorted_indices[half:][::-1]
        perm_v = list(front) + list(back)
        ms = _makespan_fast(processing_times, perm_v)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm_v
    
    # Local search on best initial solution
    if time.time() < deadline:
        perm_ls = _insertion_local_search(processing_times, best_perm, max_iter=30)
        ms = _makespan_fast(processing_times, perm_ls)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm_ls
    
    # Multiple restarts with perturbations
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Perturb the best solution
        perm_perturbed = list(best_perm)
        n_perturb = min(3, n_jobs - 1)
        for _ in range(n_perturb):
            i = rng.integers(0, n_jobs - 1)
            perm_perturbed[i], perm_perturbed[i+1] = perm_perturbed[i+1], perm_perturbed[i]
        
        # Local search from perturbed solution
        perm_ls = _insertion_local_search(processing_times, perm_perturbed, max_iter=20)
        ms = _makespan_fast(processing_times, perm_ls)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm_ls
    
    return best_perm
# EVOLVE-BLOCK-END
