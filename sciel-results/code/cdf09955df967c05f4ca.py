"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(perm, pt):
    """Compute makespan for a given permutation of job indices."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    # C[j] is completion time of job j on the current machine
    # We process machines sequentially, updating completion times
    c = np.zeros(n)
    for mi in range(m):
        # For machine mi, job perm[j] is processed in order j=0..n-1
        # Completion on machine mi for position j:
        #   C[j] = max(C[j-1], C_prev[j]) + pt[perm[j], mi]
        # where C_prev is the completion from the previous machine (or 0 for first)
        # But we need to be careful: the completion on machine mi-1 for the job at position j
        # is stored in c[j] before updating for machine mi.
        # Actually, let's use a 2D approach or careful 1D.
        # Let's use the standard recurrence:
        # C(j, k) = max(C(j-1, k), C(j, k-1)) + p(j, k)
        # We only need the last row/column. Let's do it properly.
        pass
    
    # Simpler correct approach:
    # completion[j][k] for job at position j, machine k
    # We'll use a 1D array for the current machine's completions
    # and update in-place using the previous machine's values
    
    # Let's just do the O(n*m) loop which is fine for moderate n
    # completion[j] = completion time of job at position j on the last machine processed so far
    comp = np.zeros(n)
    for mi in range(m):
        new_comp = np.zeros(n)
        for j in range(n):
            job = perm[j]
            p = pt[job, mi]
            if j == 0:
                # First job on this machine: must wait for completion on previous machine
                # which is comp[0] (from previous machine iteration)
                # But for the first machine, comp[0] is 0
                new_comp[j] = comp[j] + p
            else:
                # Must wait for: (1) previous job on this machine to finish: new_comp[j-1]
                # (2) this job to finish on previous machine: comp[j]
                new_comp[j] = max(new_comp[j-1], comp[j]) + p
        comp = new_comp
    
    return comp[-1]


def _delta_makespan_swap(perm, i, j, pt):
    """Compute the change in makespan when swapping positions i and j in perm.
    
    This is an approximation/optimization. For exact delta, we'd need to recompute
    affected parts. For simplicity and correctness, let's just recompute makespan
    for small perturbations or use a fast approximation.
    
    Actually, for SA we need exact delta. Let's compute it properly but efficiently.
    The swap affects positions i and j. All jobs between i and j are shifted.
    This is complex. Let's just recompute the full makespan for the candidate.
    For the sizes we expect, this should be fast enough.
    """
    # Swap i and j
    perm[i], perm[j] = perm[j], perm[i]
    new_ms = _makespan(perm, pt)
    # Swap back
    perm[i], perm[j] = perm[j], perm[i]
    return new_ms


def _neh(processing_times):
    """NEH heuristic: insert jobs one by one in order of decreasing total processing time."""
    n, m = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    perm = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow-shop instance."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    processing_times = np.asarray(processing_times, dtype=float)
    n, m = processing_times.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # Start with NEH heuristic
    perm = _neh(processing_times)
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    
    # Simulated Annealing
    random.seed(42)
    
    # Initial temperature based on problem size
    T = best_ms * 0.1
    T_min = best_ms * 1e-6
    cooling_rate = 0.999
    
    # Pre-compute for speed
    current_perm = list(perm)
    current_ms = best_ms
    
    # Adaptive number of iterations per temperature step
    base_steps = max(10, n * m)
    
    while T > T_min:
        if time.time() > deadline:
            break
        
        # Number of swaps at this temperature
        steps = base_steps
        for _ in range(steps):
            if time.time() > deadline:
                break
            
            # Random swap
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            # Perform swap
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            new_ms = _makespan(current_perm, processing_times)
            delta = new_ms - current_ms
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                current_ms = new_ms
                if new_ms < best_ms:
                    best_ms = new_ms
                    best_perm = list(current_perm)
            else:
                # Reject: swap back
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        
        # Cool down
        T *= cooling_rate
    
    # Final local search: 2-opt style swap optimization
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                    new_ms = _makespan(current_perm, processing_times)
                    if new_ms < current_ms:
                        current_ms = new_ms
                        improved = True
                        if new_ms < best_ms:
                            best_ms = new_ms
                            best_perm = list(current_perm)
                    else:
                        current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                if time.time() > deadline:
                    break
    
    return best_perm
# EVOLVE-BLOCK-END
