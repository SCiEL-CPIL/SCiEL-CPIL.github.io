"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, pt):
    """Compute makespan for a permutation of job indices on flow-shop."""
    n_machines = pt.shape[1]
    # C[m] = completion time of current job on machine m
    # We process jobs in perm order; for each job, machine m completion depends on
    # previous machine's completion for same job and previous job's completion on same machine
    n_jobs = len(perm)
    # completion[job_idx][machine] but we only need last job's per-machine completions for next job
    # Actually for makespan we just need the final completion on last machine
    # Let's track: for each machine, the time when it becomes free
    # And for the current job, the time when it can start on each machine
    
    # Efficient computation:
    # Let C[j][m] = completion time of job j (in permutation position) on machine m
    # C[j][0] = C[j-1][0] + pt[perm[j], 0]
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + pt[perm[j], m]
    # Makespan = C[n-1][n_machines-1]
    
    if n_machines == 0:
        return 0.0
    
    # Use a 1D array for previous job's completions per machine
    # and compute current job's completions
    prev = np.zeros(n_machines)  # C[j-1][m]
    for j in range(n_jobs):
        job = perm[j]
        curr = np.empty(n_machines)
        curr[0] = prev[0] + pt[job, 0]
        for m in range(1, n_machines):
            curr[m] = max(curr[m-1], prev[m]) + pt[job, m]
        prev = curr
    return prev[-1]


def _neh_heuristic(pt):
    """NEH (Nawaz, Enscore, Ham) heuristic for flow shop."""
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    # Sort jobs by total processing time descending
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Insert each job into best position
    perm = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _local_search_insertion(perm, pt, deadline=None):
    """Insertion-based local search. Try moving each job to every other position."""
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    current_ms = _makespan(perm, pt)
    best_perm = list(perm)
    best_ms = current_ms
    improved = True
    
    while improved:
        improved = False
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
            job = best_perm[i]
            for j in range(n_jobs):
                if i == j:
                    continue
                # Remove job from i, insert at j
                candidate = best_perm[:i] + best_perm[i+1:]
                # Adjust insertion position
                if j > i:
                    j -= 1
                candidate = candidate[:j] + [job] + candidate[j:]
                ms = _makespan(candidate, pt)
                if ms < best_ms - 1e-12:
                    best_ms = ms
                    best_perm = candidate
                    improved = True
                    # Restart from current best
                    break
            if deadline is not None and time.time() > deadline:
                break
        if deadline is not None and time.time() > deadline:
            break
    
    return best_perm, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Start with NEH heuristic
    best_perm = _neh_heuristic(pt)
    best_ms = _makespan(best_perm, pt)
    
    # Local search with insertion moves
    if time.time() < deadline - 5.0:
        best_perm, best_ms = _local_search_insertion(best_perm, pt, deadline)
    
    # If we have time, try multiple restarts with different initial heuristics
    if time.time() < deadline - 10.0 and n_jobs > 5:
        # Try SPT (Shortest Processing Time) on first machine as alternative start
        spt_order = list(np.argsort(pt[:, 0]))
        if spt_order != best_perm:
            alt_perm = list(spt_order)
            alt_ms = _makespan(alt_perm, pt)
            if time.time() < deadline - 5.0:
                alt_perm, alt_ms = _local_search_insertion(alt_perm, pt, deadline)
            if alt_ms < best_ms - 1e-12:
                best_perm = alt_perm
                best_ms = alt_ms
        
        # Try LPT on last machine
        lpt_last = list(np.argsort(-pt[:, -1]))
        if lpt_last != best_perm:
            alt_perm = list(lpt_last)
            alt_ms = _makespan(alt_perm, pt)
            if time.time() < deadline - 5.0:
                alt_perm, alt_ms = _local_search_insertion(alt_perm, pt, deadline)
            if alt_ms < best_ms - 1e-12:
                best_perm = alt_perm
                best_ms = alt_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
