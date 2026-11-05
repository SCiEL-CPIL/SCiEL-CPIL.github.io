# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, pt):
    """Compute makespan for a given permutation."""
    n = len(perm)
    m = pt.shape[1]
    completion = np.zeros(m)
    for j in perm:
        for i in range(m):
            completion[i] = completion[i] + pt[j, i]
    return completion[-1]


def _makespan_fast(perm, pt):
    """Vectorized makespan computation."""
    # perm is array of job indices
    # pt is (n_jobs, n_machines)
    # For each job in order, completion[m] = completion[m-1] + pt[j, m]
    # We can compute this iteratively but vectorize over jobs isn't straightforward
    # Let's just do it efficiently in a loop with numpy
    n_machines = pt.shape[1]
    completion = np.zeros(n_machines)
    for j in perm:
        # completion[i] = completion[i-1] + pt[j, i] for i in 0..m-1
        # completion[0] = pt[j, 0]
        # completion[i] = completion[i-1] + pt[j, i]
        completion[0] = pt[j, 0]
        for i in range(1, n_machines):
            completion[i] = completion[i-1] + pt[j, i]
    return completion[-1]


def _neh(pt):
    """NEH heuristic: insert jobs one by one in order of decreasing total processing time."""
    n, m = pt.shape
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = [order[0]]
    for k in range(1, n):
        j = order[k]
        best_pos = 0
        best_ms = _makespan_fast(perm + [j], pt)
        for pos in range(1, k + 1):
            candidate = perm[:pos] + [j] + perm[pos:]
            ms = _makespan_fast(candidate, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [j] + perm[best_pos:]
    return perm


def _palmer_pt(perm, pt):
    """Compute Palmer's priority value for inserting job at position."""
    n, m = pt.shape
    j = perm[0]
    total = pt[j].sum()
    # Palmer: sum over machines of (i * p[j][i] - (n-i) * p[j][i]) ... actually it's a bit different
    # Let's use a simpler heuristic for initial ordering
    return total


def _local_search(perm, pt, deadline):
    """Adjacent swap local search."""
    perm = list(perm)
    n = len(perm)
    current_ms = _makespan_fast(perm, pt)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            if time.time() >= deadline:
                break
            # Try swapping perm[i] and perm[i+1]
            new_perm = perm[:]
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            new_ms = _makespan_fast(new_perm, pt)
            if new_ms < current_ms:
                perm = new_perm
                current_ms = new_ms
                improved = True
    return perm, current_ms


def _segment_reversal(perm, pt, deadline):
    """Local search with segment reversal."""
    perm = list(perm)
    n = len(perm)
    current_ms = _makespan_fast(perm, pt)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    break
                new_perm = perm[:i] + perm[i:j+1][::-1] + perm[j+1:]
                new_ms = _makespan_fast(new_perm, pt)
                if new_ms < current_ms:
                    perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break
    return perm, current_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    pt = np.asarray(processing_times)
    n, m = pt.shape
    
    if n <= 1:
        return list(range(n))
    
    # Generate multiple initial solutions
    candidates = []
    
    # 1. NEH heuristic
    candidates.append(_neh(pt))
    
    # 2. LPT (Longest Processing Time)
    candidates.append(list(np.argsort(-pt.sum(axis=1))))
    
    # 3. SPT (Shortest Processing Time)
    candidates.append(list(np.argsort(pt.sum(axis=1))))
    
    # 4. Random permutations
    rng = np.random.default_rng(42)
    for _ in range(3):
        candidates.append(list(rng.permutation(n)))
    
    # Run local search on each candidate
    best_perm = None
    best_ms = float('inf')
    
    for cand in candidates:
        if time.time() >= deadline:
            break
        perm, ms = _local_search(cand, pt, deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # If we have time, try segment reversal on the best
    if time.time() < deadline and best_perm is not None:
        perm, ms = _segment_reversal(best_perm, pt, deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # If still have time, do another round of local search on best
    if time.time() < deadline and best_perm is not None:
        perm, ms = _local_search(best_perm, pt, deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    if best_perm is None:
        return list(range(n))
    
    return best_perm
# EVOLVE-BLOCK-END
