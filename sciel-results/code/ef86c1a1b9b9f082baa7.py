# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan_fast(perm, pt):
    """Compute makespan using list-based approach for speed."""
    n = len(perm)
    m = pt.shape[1]
    if m == 0 or n == 0:
        return 0.0
    
    prev = [0.0] * n
    s = 0.0
    for i in range(n):
        s += pt[perm[i], 0]
        prev[i] = s
    
    for k in range(1, m):
        curr = [0.0] * n
        curr[0] = prev[0] + pt[perm[0], k]
        for i in range(1, n):
            v1 = prev[i]
            v2 = curr[i - 1]
            curr[i] = (v1 if v1 > v2 else v2) + pt[perm[i], k]
        prev = curr
    
    return prev[-1]


def _neh_greedy(pt, sort_key):
    """NEH heuristic with a given sorting key. Returns permutation array."""
    n, m = pt.shape
    order = np.argsort(-sort_key)
    perm = []
    
    for job in order:
        if not perm:
            perm = [int(job)]
            continue
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [int(job)] + perm[pos:]
            ms = _makespan_fast(np.array(new_perm, dtype=np.int64), pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, int(job))
    
    return np.array(perm, dtype=np.int64)


def _local_search_2opt(perm, pt, max_sweeps=200, deadline=None):
    """First-improvement 2-opt local search."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = perm.copy()
    best_ms = _makespan_fast(best_perm, pt)
    
    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if deadline is not None and time.time() > deadline:
                return best_perm
            for j in range(i + 1, n):
                new_perm = best_perm.copy()
                new_perm[i:j + 1] = best_perm[i:j + 1][::-1]
                ms = _makespan_fast(new_perm, pt)
                if ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_perm


def _local_search_swap(perm, pt, max_sweeps=100, deadline=None):
    """First-improvement swap local search."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = perm.copy()
    best_ms = _makespan_fast(best_perm, pt)
    
    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if deadline is not None and time.time() > deadline:
                return best_perm
            for j in range(i + 1, n):
                new_perm = best_perm.copy()
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                ms = _makespan_fast(new_perm, pt)
                if ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_perm


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    best_perm = None
    best_ms = float('inf')
    
    # Try multiple NEH variants
    variants = []
    totals = pt.sum(axis=1)
    avgs = pt.mean(axis=1)
    mins = pt.min(axis=1)
    maxs = pt.max(axis=1)
    
    for key in [totals, avgs, mins, maxs]:
        if time.time() >= deadline:
            break
        perm = _neh_greedy(pt, key)
        ms = _makespan_fast(perm, pt)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    if best_perm is None:
        return list(range(n))
    
    # Local search from best NEH
    if time.time() < deadline:
        ls_perm = _local_search_2opt(best_perm, pt, max_sweeps=300, deadline=deadline)
        ls_ms = _makespan_fast(ls_perm, pt)
        if ls_ms < best_ms:
            best_perm = ls_perm
            best_ms = ls_ms
    
    # Try local search from other NEH variants
    for key in [totals, avgs, mins, maxs]:
        if time.time() >= deadline:
            break
        perm = _neh_greedy(pt, key)
        ms = _makespan_fast(perm, pt)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
        if time.time() < deadline:
            ls_perm = _local_search_2opt(perm, pt, max_sweeps=150, deadline=deadline)
            ls_ms = _makespan_fast(ls_perm, pt)
            if ls_ms < best_ms:
                best_perm = ls_perm
                best_ms = ls_ms
    
    # Random restarts with swap local search
    rng = np.random.default_rng(42)
    if time.time() < deadline:
        remaining = deadline - time.time()
        num_restarts = min(10, max(1, int(remaining / 5)))
        for r in range(num_restarts):
            if time.time() >= deadline:
                break
            random_perm = rng.permutation(n).astype(np.int64)
            ls_r = _local_search_swap(random_perm, pt, max_sweeps=80, deadline=deadline)
            ls_ms_r = _makespan_fast(ls_r, pt)
            if ls_ms_r < best_ms:
                best_perm = ls_r
                best_ms = ls_ms_r
            # Also try 2-opt from random if time permits
            if time.time() < deadline - 2:
                ls_2opt_r = _local_search_2opt(random_perm, pt, max_sweeps=50, deadline=deadline)
                ms_2opt_r = _makespan_fast(ls_2opt_r, pt)
                if ms_2opt_r < best_ms:
                    best_perm = ls_2opt_r
                    best_ms = ms_2opt_r
    
    return list(best_perm)
# EVOLVE-BLOCK-END
