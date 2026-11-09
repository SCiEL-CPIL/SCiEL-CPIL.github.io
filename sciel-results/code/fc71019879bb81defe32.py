# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan_fast(perm, pt):
    """Optimized makespan computation using pure Python loops for small n."""
    n = len(perm)
    m = pt.shape[1]
    if m == 0 or n == 0:
        return 0.0
    
    # Use list-based approach for better cache performance
    prev = [0.0] * n
    # Machine 0
    s = 0.0
    for i in range(n):
        s += pt[perm[i], 0]
        prev[i] = s
    
    for k in range(1, m):
        curr = [0.0] * n
        curr[0] = prev[0] + pt[perm[0], k]
        for i in range(1, n):
            v1 = prev[i]
            v2 = curr[i-1]
            curr[i] = (v1 if v1 > v2 else v2) + pt[perm[i], k]
        prev = curr
    
    return prev[-1]


def _completion_matrix(perm, pt):
    """Compute full completion matrix C[i][k] = completion of job i on machine k.
    Returns a list of lists of floats, shape (n, m)."""
    n = len(perm)
    m = pt.shape[1]
    if n == 0 or m == 0:
        return []
    
    C = [[0.0] * m for _ in range(n)]
    # Machine 0
    s = 0.0
    for i in range(n):
        s += pt[perm[i], 0]
        C[i][0] = s
    
    for k in range(1, m):
        for i in range(n):
            if i == 0:
                C[i][k] = C[i][k-1] + pt[perm[i], k]
            else:
                v1 = C[i][k-1]
                v2 = C[i-1][k]
                C[i][k] = (v1 if v1 > v2 else v2) + pt[perm[i], k]
    
    return C


def _makespan_from_C(C, n, m):
    """Extract makespan from completion matrix."""
    if n == 0 or m == 0:
        return 0.0
    return C[n-1][m-1]


def _reversal_delta(perm, C, i, j, pt, m):
    """Compute new makespan after reversing segment [i..j] in perm.
    Uses the completion matrix C for the current perm.
    Returns the new makespan (float).
    
    The reversal only affects completions in positions i..j.
    For positions < i: unchanged.
    For positions > j: need to recompute from position j onward, but we can
    bound: the makespan is C_new[n-1][m-1].
    
    Strategy: recompute completions for the reversed segment and everything after.
    """
    n = len(perm)
    
    # Build new perm with reversal
    # Positions i..j are reversed
    # We need C_new for all positions, but only positions i..n-1 change.
    
    # For efficiency, copy C for positions < i
    C_new = [row[:] for row in C[:i]]
    
    # Positions i..j are reversed: new position p gets old position (i+j-p)
    # We need to recompute completions for these positions.
    # The input to position i is C_new[i-1] (or None if i==0).
    # For each machine k, the completion at position p depends on:
    #   C_new[p][k] = max(C_new[p][k-1], C_new[p-1][k]) + pt[perm_new[p]][k]
    
    perm_new = list(perm)
    perm_new[i:j+1] = perm_new[i:j+1][::-1]
    
    # Recompute for positions i through n-1
    for p in range(i, n):
        row = [0.0] * m
        for k in range(m):
            if k == 0:
                row[k] = (C_new[p-1][0] if p > 0 else 0.0) + pt[perm_new[p], 0]
            else:
                v1 = row[k-1]
                v2 = C_new[p-1][k] if p > 0 else 0.0
                row[k] = (v1 if v1 > v2 else v2) + pt[perm_new[p], k]
        C_new.append(row)
    
    return C_new[n-1][m-1]


def _neh_heuristic(pt):
    """NEH heuristic for flow shop. Returns permutation."""
    n, m = pt.shape
    
    # Sort jobs by total processing time in descending order
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    # Build permutation greedily
    perm = []
    
    for job in order:
        if not perm:
            perm = [job]
            continue
        
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(np.array(new_perm, dtype=np.int64), pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm.insert(best_pos, job)
    
    return np.array(perm, dtype=np.int64)


def _local_search_2opt(perm, pt, max_iters=500, deadline=None):
    """2-opt local search for flow shop with time limit, using incremental makespan."""
    n = len(perm)
    if n <= 1:
        return perm
    
    m = pt.shape[1]
    best_perm = perm.copy()
    C = _completion_matrix(best_perm, pt)
    best_ms = C[n-1][m-1]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n - 1):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(i + 1, n):
                if deadline is not None and time.time() > deadline:
                    break
                new_ms = _reversal_delta(best_perm, C, i, j, pt, m)
                if new_ms < best_ms - 1e-9:
                    # Apply reversal
                    best_perm[i:j+1] = best_perm[i:j+1][::-1]
                    C = _completion_matrix(best_perm, pt)
                    best_ms = new_ms
                    improved = True
    
    return best_perm


def _local_search_swap(perm, pt, max_iters=200, deadline=None):
    """Swap-based local search for flow shop with incremental makespan."""
    n = len(perm)
    if n <= 1:
        return perm
    
    m = pt.shape[1]
    best_perm = perm.copy()
    C = _completion_matrix(best_perm, pt)
    best_ms = C[n-1][m-1]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n - 1):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(i + 1, n):
                if deadline is not None and time.time() > deadline:
                    break
                # Swap positions i and j
                new_perm = list(best_perm)
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                # Recompute completions from min(i,j) onward
                p_start = min(i, j)
                C_new = [row[:] for row in C[:p_start]]
                for p in range(p_start, n):
                    row = [0.0] * m
                    for k in range(m):
                        if k == 0:
                            row[k] = (C_new[p-1][0] if p > 0 else 0.0) + pt[new_perm[p], 0]
                        else:
                            v1 = row[k-1]
                            v2 = C_new[p-1][k] if p > 0 else 0.0
                            row[k] = (v1 if v1 > v2 else v2) + pt[new_perm[p], k]
                    C_new.append(row)
                new_ms = C_new[n-1][m-1]
                if new_ms < best_ms - 1e-9:
                    best_perm = np.array(new_perm, dtype=np.int64)
                    C = C_new
                    best_ms = new_ms
                    improved = True
    
    return best_perm


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # Start with NEH heuristic
    best_perm = _neh_heuristic(pt)
    best_ms = _makespan_fast(best_perm, pt)
    
    # Local search with 2-opt
    if time.time() < deadline:
        ls_perm = _local_search_2opt(best_perm, pt, max_iters=300, deadline=deadline)
        ls_ms = _makespan_fast(ls_perm, pt)
        if ls_ms < best_ms:
            best_perm = ls_perm
            best_ms = ls_ms
    
    # Try NEH with average processing time
    if time.time() < deadline:
        avgs = pt.mean(axis=1)
        order2 = np.argsort(-avgs)
        perm2 = []
        for job in order2:
            if not perm2:
                perm2 = [job]
                continue
            best_pos = 0
            best_ms2 = float('inf')
            for pos in range(len(perm2) + 1):
                new_perm = perm2[:pos] + [job] + perm2[pos:]
                ms = _makespan_fast(np.array(new_perm, dtype=np.int64), pt)
                if ms < best_ms2:
                    best_ms2 = ms
                    best_pos = pos
            perm2.insert(best_pos, job)
        perm2_arr = np.array(perm2, dtype=np.int64)
        ms2 = _makespan_fast(perm2_arr, pt)
        if ms2 < best_ms:
            best_perm = perm2_arr
            best_ms = ms2
            if time.time() < deadline:
                ls_perm2 = _local_search_2opt(best_perm, pt, max_iters=200, deadline=deadline)
                ls_ms2 = _makespan_fast(ls_perm2, pt)
                if ls_ms2 < best_ms:
                    best_perm = ls_perm2
                    best_ms = ls_ms2
    
    # Try NEH with minimum processing time
    if time.time() < deadline:
        mins = pt.min(axis=1)
        order3 = np.argsort(-mins)
        perm3 = []
        for job in order3:
            if not perm3:
                perm3 = [job]
                continue
            best_pos = 0
            best_ms3 = float('inf')
            for pos in range(len(perm3) + 1):
                new_perm = perm3[:pos] + [job] + perm3[pos:]
                ms = _makespan_fast(np.array(new_perm, dtype=np.int64), pt)
                if ms < best_ms3:
                    best_ms3 = ms
                    best_pos = pos
            perm3.insert(best_pos, job)
        perm3_arr = np.array(perm3, dtype=np.int64)
        ms3 = _makespan_fast(perm3_arr, pt)
        if ms3 < best_ms:
            best_perm = perm3_arr
            best_ms = ms3
            if time.time() < deadline:
                ls_perm3 = _local_search_2opt(best_perm, pt, max_iters=200, deadline=deadline)
                ls_ms3 = _makespan_fast(ls_perm3, pt)
                if ls_ms3 < best_ms:
                    best_perm = ls_perm3
                    best_ms = ls_ms3
    
    # Random restarts with local search
    rng = np.random.default_rng(42)
    if time.time() < deadline:
        num_restarts = 3
        for r in range(num_restarts):
            if time.time() >= deadline:
                break
            random_perm = rng.permutation(n).astype(np.int64)
            # Quick local search from random start
            ls_r = _local_search_swap(random_perm, pt, max_iters=100, deadline=deadline)
            ls_ms_r = _makespan_fast(ls_r, pt)
            if ls_ms_r < best_ms:
                best_perm = ls_r
                best_ms = ls_ms_r
            # If improved, try 2-opt as well
            if time.time() < deadline:
                ls_r2 = _local_search_2opt(ls_r, pt, max_iters=100, deadline=deadline)
                ls_ms_r2 = _makespan_fast(ls_r2, pt)
                if ls_ms_r2 < best_ms:
                    best_perm = ls_r2
                    best_ms = ls_ms_r2
    
    # Final polish: alternate 2-opt and swap with increased iterations
    if time.time() < deadline:
        for polish_round in range(3):
            if time.time() >= deadline:
                break
            # 2-opt polish
            if time.time() < deadline:
                p2 = _local_search_2opt(best_perm, pt, max_iters=150, deadline=deadline)
                C_p2 = _completion_matrix(p2, pt)
                ms_p2 = C_p2[len(p2)-1][pt.shape[1]-1] if len(p2) > 0 else 0.0
                if ms_p2 < best_ms:
                    best_perm = p2
                    best_ms = ms_p2
            # Swap polish
            if time.time() < deadline:
                ps = _local_search_swap(best_perm, pt, max_iters=100, deadline=deadline)
                C_ps = _completion_matrix(ps, pt)
                ms_ps = C_ps[len(ps)-1][pt.shape[1]-1] if len(ps) > 0 else 0.0
                if ms_ps < best_ms:
                    best_perm = ps
                    best_ms = ms_ps
    
    return list(best_perm)
# EVOLVE-BLOCK-END
