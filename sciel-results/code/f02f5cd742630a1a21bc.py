"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def order_jobs(processing_times):
    # Explicit deadline guard: 120s limit, keep safety margin of 5s
    start_time = time.time()
    deadline = start_time + 115.0
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
        
    # Helper: Calculate makespan for a given permutation
    # O(n_jobs * n_machines)
    def calc_makespan(perm):
        if len(perm) == 0:
            return 0.0
        # pt[perm] gives processing times in order of jobs
        # Cumulative sum along machines axis
        # Shape: (n_jobs, n_machines)
        # We need to track the completion time of the last machine for each job
        # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
        
        # Vectorized approach for speed:
        # This is a standard DP. 
        # C[:, 0] = cumsum(P[:, 0])
        # For m > 0: C[:, m] = cumsum along jobs of max(C[:, m-1], C[:, m-1])? No.
        # C[j, m] depends on C[j-1, m] and C[j, m-1].
        # This is a prefix-max-plus convolution which isn't directly vectorizable in a simple single loop without iteration over jobs or machines.
        # Given n_jobs is likely small (synthetic instances), a loop over jobs is fine.
        
        # Let's use a loop over jobs, vectorizing over machines if possible?
        # Actually, for small n_jobs, a simple loop is often faster than complex vectorization overhead.
        # Let's try a loop over jobs.
        
        C_prev = np.zeros(n_machines)
        for j in perm:
            p_j = pt[j]
            # C_curr[m] = max(C_prev[m], C_curr[m-1]) + p_j[m]
            # This still has a dependency along m.
            # If n_machines is small, loop over m too.
            # If n_jobs is large, we need speed.
            
            # Let's assume n_jobs and n_machines are moderate (e.g. < 100).
            # Double loop is O(N*M).
            C_curr = np.zeros(n_machines)
            for m in range(n_machines):
                if m == 0:
                    val = C_prev[0] + p_j[0]
                else:
                    val = max(C_prev[m], C_curr[m-1]) + p_j[m]
                C_curr[m] = val
            C_prev = C_curr
        return C_prev[-1]

    # Initial Heuristic: NEH (Nawaz, Enscore, Ham)
    # 1. Sort jobs by total processing time descending
    # 2. Insert each job into the position that minimizes makespan
    
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = []
    for j in order:
        best_pos = 0
        best_ms = np.inf
        # Try all positions
        # Optimization: If perm is small, this is fast.
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [j] + perm[pos:]
            ms = calc_makespan(new_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, j)
        
    current_perm = np.array(perm)
    current_ms = calc_makespan(current_perm)
    best_perm = current_perm.copy()
    best_ms = current_ms
    
    # ILS Loop
    # Time check inside loop
    while time.time() < deadline:
        # Perturbation: Random swap of two jobs
        i, j = np.random.choice(n_jobs, 2, replace=False)
        
        # Apply swap
        perm_new = current_perm.copy()
        perm_new[i], perm_new[j] = perm_new[j], perm_new[i]
        
        ms_new = calc_makespan(perm_new)
        
        if ms_new < current_ms:
            current_ms = ms_new
            current_perm = perm_new
            if ms_new < best_ms:
                best_ms = ms_new
                best_perm = current_perm.copy()
        else:
            # Accept with some probability to escape local optima? 
            # Simple ILS usually accepts if better, or uses simulated annealing.
            # Given the short time, let's stick to steepest descent with random restarts if stuck?
            # Or simple random walk acceptance?
            # Let's try a simple random acceptance to explore, but keep best.
            # Actually, for makespan, random swaps often make it worse.
            # Let's try 2-opt moves (reverse segment) which are more effective for TSP-like structures, though flow shop is different.
            # Let's stick to swaps but add a small chance to accept worse solutions (Simulated Annealing lite).
            # T decreases over time.
            elapsed = time.time() - start_time
            T = max(0.01, 1.0 * (1.0 - elapsed / 115.0))
            if np.random.rand() < np.exp((current_ms - ms_new) / T):
                current_ms = ms_new
                current_perm = perm_new
            
    return list(best_perm.astype(int))

# EVOLVE-BLOCK-END
