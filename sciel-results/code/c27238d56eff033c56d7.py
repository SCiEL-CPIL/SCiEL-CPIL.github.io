# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan_fast(perm, pt):
    """Optimized makespan computation using pure Python loops for small n."""
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
            v2 = curr[i-1]
            curr[i] = (v1 if v1 > v2 else v2) + pt[perm[i], k]
        prev = curr
    
    return prev[-1]


def _completion_matrix(perm, pt):
    """Compute full completion matrix C[i][k] = completion of job i on machine k."""
    n = len(perm)
    m = pt.shape[1]
    if n == 0 or m == 0:
        return []
    
    C = [[0.0] * m for _ in range(n)]
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


def _reversal_delta(perm, C, i, j, pt, m):
    """Compute new makespan after reversing segment [i..j] in perm using completion matrix."""
    n = len(perm)
    
    C_new = [row[:] for row in C[:i]]
    
    perm_new = list(perm)
    perm_new[i:j+1] = perm_new[i:j+1][::-1]
    
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
    
    return C_new[n-1][m-1], C_new, perm_new


def _swap_delta(perm, C, i, j, pt, m):
    """Compute new makespan after swapping positions i and j using completion matrix."""
    n = len(perm)
    p_start = min(i, j)
    
    C_new = [row[:] for row in C[:p_start]]
    perm_new = list(perm)
    perm_new[i], perm_new[j] = perm_new[j], perm_new[i]
    
    for p in range(p_start, n):
        row = [0.0] * m
        for k in range(m):
            if k == 0:
                row[k] = (C_new[p-1][0] if p > 0 else 0.0) + pt[perm_new[p], 0]
            else:
                v1 = row[k-1]
                v2 = C_new[p-1][k] if p > 0 else 0.0
                row[k] = (v1 if v1 > v2 else v2) + pt[perm_new[p], k]
        C_new.append(row)
    
    return C_new[n-1][m-1], C_new, perm_new


def _insert_delta(perm, C, i, j, pt, m):
    """Compute new makespan after moving job from position i to position j (0-indexed, j != i).
    If i < j, job moves right; if i > j, job moves left.
    j is the insertion position in the remaining sequence (before re-inserting).
    Returns (new_makespan, C_new, perm_new) or None if invalid."""
    n = len(perm)
    if i == j:
        return None
    
    # Remove job at position i
    job = perm[i]
    remaining = list(perm[:i]) + list(perm[i+1:])
    
    # Insert at position j in remaining
    # If i < j, j in remaining corresponds to original position j
    # If i > j, j in remaining is the same
    perm_new = remaining[:j] + [job] + remaining[j:]
    
    p_start = min(i, j)
    
    C_new = [row[:] for row in C[:p_start]]
    
    for p in range(p_start, n):
        row = [0.0] * m
        for k in range(m):
            if k == 0:
                row[k] = (C_new[p-1][0] if p > 0 else 0.0) + pt[perm_new[p], 0]
            else:
                v1 = row[k-1]
                v2 = C_new[p-1][k] if p > 0 else 0.0
                row[k] = (v1 if v1 > v2 else v2) + pt[perm_new[p], k]
        C_new.append(row)
    
    return C_new[n-1][m-1], C_new, perm_new


def _neh_heuristic(pt, sort_key='sum'):
    """NEH heuristic for flow shop. Returns permutation."""
    n, m = pt.shape
    
    if sort_key == 'sum':
        totals = pt.sum(axis=1)
    elif sort_key == 'mean':
        totals = pt.mean(axis=1)
    elif sort_key == 'min':
        totals = pt.min(axis=1)
    elif sort_key == 'max':
        totals = pt.max(axis=1)
    else:
        totals = pt.sum(axis=1)
    
    order = np.argsort(-totals)
    
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


def _local_search_2opt(perm, pt, max_iters=300, deadline=None):
    """2-opt local search using incremental completion matrix."""
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
                return best_perm
            for j in range(i + 1, n):
                if deadline is not None and time.time() > deadline:
                    return best_perm
                new_ms, C_new, perm_new = _reversal_delta(best_perm, C, i, j, pt, m)
                if new_ms < best_ms - 1e-9:
                    best_perm = np.array(perm_new, dtype=np.int64)
                    C = C_new
                    best_ms = new_ms
                    improved = True
    
    return best_perm


def _local_search_swap(perm, pt, max_iters=200, deadline=None):
    """Swap-based local search using incremental completion matrix."""
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
                return best_perm
            for j in range(i + 1, n):
                if deadline is not None and time.time() > deadline:
                    return best_perm
                new_ms, C_new, perm_new = _swap_delta(best_perm, C, i, j, pt, m)
                if new_ms < best_ms - 1e-9:
                    best_perm = np.array(perm_new, dtype=np.int64)
                    C = C_new
                    best_ms = new_ms
                    improved = True
    
    return best_perm


def _local_search_insert(perm, pt, max_iters=100, deadline=None):
    """Insertion-based local search: move one job to a different position."""
    n = len(perm)
    if n <= 2:
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
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                return best_perm
            for j in range(n):
                if i == j:
                    continue
                if deadline is not None and time.time() > deadline:
                    return best_perm
                result = _insert_delta(best_perm, C, i, j, pt, m)
                if result is None:
                    continue
                new_ms, C_new, perm_new = result
                if new_ms < best_ms - 1e-9:
                    best_perm = np.array(perm_new, dtype=np.int64)
                    C = C_new
                    best_ms = new_ms
                    improved = True
                    break  # Restart scan after improvement
    
    return best_perm


def _tabu_search(perm, pt, deadline, tabu_tenure=12, max_iters=2000):
    """Tabu search with insertion moves using fast makespan."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = perm.copy()
    best_ms = _makespan_fast(best_perm, pt)
    current_perm = perm.copy()
    current_ms = best_ms
    
    # Tabu list: (from_pos, to_pos) -> iteration when it was made tabu
    tabu_list = {}
    tabu_iter = 0
    
    for it in range(max_iters):
        if time.time() >= deadline:
            break
        
        # Periodically clean tabu list
        if tabu_iter % 300 == 0 and tabu_iter > 0:
            tabu_list = {k: v for k, v in tabu_list.items() if v > tabu_iter - tabu_tenure}
        
        best_move = None
        best_move_ms = float('inf')
        
        for i in range(n):
            if time.time() >= deadline:
                return best_perm
            job = current_perm[i]
            remaining = np.delete(current_perm, i)
            for pos in range(len(remaining) + 1):
                if pos == i:
                    continue
                new_perm = np.insert(remaining, pos, job)
                ms = _makespan_fast(new_perm, pt)
                
                move_key = (i, pos)
                is_tabu = tabu_list.get(move_key, -1) >= tabu_iter - tabu_tenure
                
                # Aspiration: accept tabu move if it improves global best
                if is_tabu and ms < best_ms - 1e-9:
                    is_tabu = False
                
                if not is_tabu:
                    if ms < best_move_ms:
                        best_move_ms = ms
                        best_move = (i, pos)
        
        if best_move is None:
            break
        
        i, pos = best_move
        job = current_perm[i]
        remaining = np.delete(current_perm, i)
        current_perm = np.insert(remaining, pos, job)
        current_ms = best_move_ms
        
        # Update tabu list
        tabu_list[(i, pos)] = tabu_iter
        # Reverse move
        reverse_pos = pos if pos > i else pos
        tabu_list[(pos, i)] = tabu_iter
        
        tabu_iter += 1
        
        if current_ms < best_ms - 1e-9:
            best_ms = current_ms
            best_perm = current_perm.copy()
    
    return best_perm


def _simulated_annealing(perm, pt, deadline, max_iters=3000):
    """Simulated annealing for flow shop."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = perm.copy()
    best_ms = _makespan_fast(best_perm, pt)
    current_perm = perm.copy()
    current_ms = best_ms
    
    T = max(current_ms * 0.1, 1.0)
    cooling_rate = 0.995
    rng = np.random.default_rng(123)
    
    for it in range(max_iters):
        if time.time() >= deadline:
            break
        
        # Random swap perturbation
        i, j = rng.choice(n, 2, replace=False)
        new_perm = current_perm.copy()
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        new_ms = _makespan_fast(new_perm, pt)
        
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms - 1e-9:
                best_ms = current_ms
                best_perm = current_perm.copy()
        
        T *= cooling_rate
        if T < 1e-6:
            T = 1e-6
    
    return best_perm


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # Start with NEH heuristic (multiple sort keys)
    best_perm = _neh_heuristic(pt, 'sum')
    best_ms = _makespan_fast(best_perm, pt)
    
    # Try other NEH variants
    for key in ['mean', 'min', 'max']:
        if time.time() >= deadline:
            break
        perm_var = _neh_heuristic(pt, key)
        ms_var = _makespan_fast(perm_var, pt)
        if ms_var < best_ms:
            best_perm = perm_var
            best_ms = ms_var
    
    # Phase 1: Intensive local search from best NEH
    if time.time() < deadline:
        ls_perm = _local_search_2opt(best_perm, pt, max_iters=200, deadline=deadline)
        ls_ms = _makespan_fast(ls_perm, pt)
        if ls_ms < best_ms:
            best_perm = ls_perm
            best_ms = ls_ms
    
    if time.time() < deadline:
        ls_perm = _local_search_swap(best_perm, pt, max_iters=150, deadline=deadline)
        ls_ms = _makespan_fast(ls_perm, pt)
        if ls_ms < best_ms:
            best_perm = ls_perm
            best_ms = ls_ms
    
    if time.time() < deadline:
        ls_perm = _local_search_insert(best_perm, pt, max_iters=80, deadline=deadline)
        ls_ms = _makespan_fast(ls_perm, pt)
        if ls_ms < best_ms:
            best_perm = ls_perm
            best_ms = ls_ms
    
    # Phase 2: Tabu search to escape local optima
    if time.time() < deadline:
        tabu_perm = _tabu_search(best_perm, pt, deadline, tabu_tenure=12, max_iters=1500)
        tabu_ms = _makespan_fast(tabu_perm, pt)
        if tabu_ms < best_ms:
            best_perm = tabu_perm
            best_ms = tabu_ms
            # Polish after tabu
            if time.time() < deadline:
                p2 = _local_search_2opt(best_perm, pt, max_iters=100, deadline=deadline)
                ms2 = _makespan_fast(p2, pt)
                if ms2 < best_ms:
                    best_perm = p2
                    best_ms = ms2
    
    # Phase 3: Simulated annealing for further escape
    if time.time() < deadline:
        sa_perm = _simulated_annealing(best_perm, pt, deadline, max_iters=2000)
        sa_ms = _makespan_fast(sa_perm, pt)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
            # Polish after SA
            if time.time() < deadline:
                p2 = _local_search_swap(best_perm, pt, max_iters=100, deadline=deadline)
                ms2 = _makespan_fast(p2, pt)
                if ms2 < best_ms:
                    best_perm = p2
                    best_ms = ms2
    
    # Phase 4: Random restarts with local search
    rng = np.random.default_rng(42)
    if time.time() < deadline:
        num_restarts = 3
        for r in range(num_restarts):
            if time.time() >= deadline:
                break
            random_perm = rng.permutation(n).astype(np.int64)
            # Quick local search from random start
            ls_r = _local_search_swap(random_perm, pt, max_iters=80, deadline=deadline)
            ls_ms_r = _makespan_fast(ls_r, pt)
            if ls_ms_r < best_ms:
                best_perm = ls_r
                best_ms = ls_ms_r
            if time.time() < deadline:
                ls_r2 = _local_search_2opt(ls_r, pt, max_iters=80, deadline=deadline)
                ls_ms_r2 = _makespan_fast(ls_r2, pt)
                if ls_ms_r2 < best_ms:
                    best_perm = ls_r2
                    best_ms = ls_ms_r2
    
    # Phase 5: Final polish - alternate neighborhoods
    if time.time() < deadline:
        for polish_round in range(3):
            if time.time() >= deadline:
                break
            p2 = _local_search_2opt(best_perm, pt, max_iters=80, deadline=deadline)
            ms2 = _makespan_fast(p2, pt)
            if ms2 < best_ms:
                best_perm = p2
                best_ms = ms2
            if time.time() >= deadline:
                break
            ps = _local_search_swap(best_perm, pt, max_iters=60, deadline=deadline)
            ms_s = _makespan_fast(ps, pt)
            if ms_s < best_ms:
                best_perm = ps
                best_ms = ms_s
            if time.time() >= deadline:
                break
            pi = _local_search_insert(best_perm, pt, max_iters=40, deadline=deadline)
            ms_i = _makespan_fast(pi, pt)
            if ms_i < best_ms:
                best_perm = pi
                best_ms = ms_i
    
    return list(best_perm)
# EVOLVE-BLOCK-END
