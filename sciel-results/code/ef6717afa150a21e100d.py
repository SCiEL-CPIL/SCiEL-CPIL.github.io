"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, pt):
    """Compute makespan for a given permutation of job indices."""
    n = len(perm)
    if n == 0:
        return 0.0
    # pt is (jobs, machines)
    # perm is array of job indices in order
    # Compute cumulative times per machine
    # For each machine m, the start time of job j on machine m is max(finish on m-1, finish of previous job on m)
    # We can vectorize partially but for clarity and speed with numpy:
    # Let's build the flow shop completion times.
    
    # For efficiency, we can iterate over machines and update a vector of completion times
    # comp[j] = completion time of job j on the current machine
    # Actually, we need to track per-job completion on each machine.
    # Let's use a 1D array 'comp' where comp[j] is the completion time of job j on the previous machine.
    # Initially, comp is all 0.
    # For each machine m:
    #   For each job in order:
    #     start = max(comp[j], prev_machine_time)  # but prev_machine_time is the completion of the previous job on this machine
    #     Actually, standard flow shop:
    #     C[j, 0] = P[j, 0]
    #     C[j, m] = max(C[j, m-1], C[j-1, m]) + P[j, m]
    
    # Let's implement efficiently.
    # n jobs, m machines.
    # We'll use a 1D array 'comp' of size n, initialized to 0.
    # For each machine m:
    #   prev_job_end = 0
    #   for each job j in order:
    #       comp[j] = max(comp[j], prev_job_end) + pt[j, m]
    #       prev_job_end = comp[j]
    #   # But wait, comp[j] needs to be updated for the next machine.
    #   # Actually, we need to be careful: comp[j] is being overwritten.
    #   # Let's use a different approach: keep an array 'finish' of size n.
    #   # For machine m, we compute new_finish[j] for each job j in order.
    #   # new_finish[j] = max(finish[j], prev_job_finish) + pt[j, m]
    #   # where prev_job_finish is the finish time of the previous job on this machine.
    
    finish = np.zeros(n, dtype=np.float64)
    for m in range(pt.shape[1]):
        prev_job_finish = 0.0
        for idx in range(n):
            j = perm[idx]
            # finish[j] is the completion time of job j on machine m-1 (or 0 if m=0)
            # The start time of job j on machine m is max(finish[j], prev_job_finish)
            start = finish[j] if finish[j] > prev_job_finish else prev_job_finish
            finish[j] = start + pt[j, m]
            prev_job_finish = finish[j]
    return finish[perm[-1]]  # The last job's finish time on the last machine is the makespan


def _neh(pt):
    """NEH heuristic: insert jobs one by one in order of decreasing total processing time."""
    n, m = pt.shape
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    perm = [order[0]]
    for k in range(1, n):
        j = order[k]
        best_pos = 0
        best_mk = _makespan(np.array(perm), pt)
        # Try inserting j at each position
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [j] + perm[pos:]
            mk = _makespan(np.array(new_perm), pt)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        perm = perm[:best_pos] + [j] + perm[best_pos:]
    return perm


def _palmer(pt):
    """Palmer heuristic: sort by (n-1)*sum(P[j,m]) - sum(m*P[j,m]) or similar."""
    n, m = pt.shape
    # Palmer's rule: key = (n - 1) * sum(P[j, m]) - sum(m * P[j, m])
    # Actually, the key is often: (n-1)*sum_m P[j,m] - sum_m m*P[j,m]
    # Or: sum_m (n - m) * P[j, m]
    keys = np.zeros(n)
    for j in range(n):
        s = 0.0
        for m_idx in range(m):
            s += (n - m_idx) * pt[j, m_idx]
        keys[j] = s
    return list(np.argsort(-keys))


def _lpt(pt):
    """LPT: Longest Processing Time first (by total)."""
    return list(np.argsort(-pt.sum(axis=1)))


def _spt(pt):
    """SPT: Shortest Processing Time first (by total)."""
    return list(np.argsort(pt.sum(axis=1)))


def _random_perm(n):
    return list(np.random.permutation(n))


def _local_search(perm, pt, time_limit, rng):
    """Adjacent swap and segment reversal local search."""
    n = len(perm)
    best_perm = perm[:]
    best_mk = _makespan(np.array(best_perm), pt)
    
    # Precompute pt as numpy array for faster access
    pt_arr = np.asarray(pt)
    
    start_time = time.time()
    
    # Adjacent swap
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                break
            # Swap i and i+1
            new_perm = best_perm[:]
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            mk = _makespan(np.array(new_perm), pt_arr)
            if mk < best_mk - 1e-9:
                best_perm = new_perm
                best_mk = mk
                improved = True
        if time.time() - start_time > time_limit:
            break
    
    # Segment reversal
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit:
                    break
                new_perm = best_perm[:]
                new_perm[i:j+1] = new_perm[i:j+1][::-1]
                mk = _makespan(np.array(new_perm), pt_arr)
                if mk < best_mk - 1e-9:
                    best_perm = new_perm
                    best_mk = mk
                    improved = True
        if time.time() - start_time > time_limit:
            break
    
    return best_perm, best_mk


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on local synthetic flow-shop instances.
    
    Args:
        processing_times: np.ndarray of shape (jobs, machines)
    
    Returns:
        permutation of job indices (list of ints)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n <= 1:
        return list(range(n))
    
    rng = np.random.default_rng(42)
    
    # Generate initial solutions using multiple heuristics
    candidates = []
    
    # NEH
    candidates.append(_neh(pt))
    
    # Palmer
    candidates.append(_palmer(pt))
    
    # LPT
    candidates.append(_lpt(pt))
    
    # SPT
    candidates.append(_spt(pt))
    
    # Random restarts
    n_random = min(10, max(3, n // 2))
    for _ in range(n_random):
        candidates.append(_random_perm(n))
    
    # Evaluate all candidates and pick the best
    best_perm = None
    best_mk = float('inf')
    
    for perm in candidates:
        mk = _makespan(np.array(perm), pt)
        if mk < best_mk:
            best_mk = mk
            best_perm = perm[:]
    
    # Local search on the best candidate
    remaining_time = deadline - time.time()
    if remaining_time > 1.0:
        best_perm, best_mk = _local_search(best_perm, pt, remaining_time * 0.8, rng)
    
    # If we have time, try local search on other good candidates
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Sort candidates by makespan
        scored = []
        for perm in candidates:
            mk = _makespan(np.array(perm), pt)
            scored.append((mk, perm))
        scored.sort(key=lambda x: x[0])
        
        for mk, perm in scored[1:5]:  # Try top 5
            if time.time() > deadline:
                break
            t_left = deadline - time.time()
            if t_left < 1.0:
                break
            # Only do local search if it could potentially beat the best
            if mk < best_mk * 1.1:  # Within 10% of best
                new_perm, new_mk = _local_search(perm, pt, t_left * 0.5, rng)
                if new_mk < best_mk:
                    best_mk = new_mk
                    best_perm = new_perm
    
    return best_perm
# EVOLVE-BLOCK-END
