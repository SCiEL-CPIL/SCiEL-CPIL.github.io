# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _makespan_vec(processing_times, perm):
    """Compute makespan using vectorized NumPy."""
    n = len(perm)
    m = processing_times.shape[1]
    if n == 0 or m == 0:
        return 0.0
    # C[j][k] = max(C[j][k-1], C[j-1][k]) + p[perm[j]][k]
    # Vectorize over machines for each job
    P = processing_times[perm]  # (n, m)
    C = np.zeros((n, m))
    C[0] = np.cumsum(P[0])
    for j in range(1, n):
        prev = C[j-1]
        cur = np.maximum(prev, np.cumsum(P[j], axis=0) + prev[0] - np.cumsum(P[j], axis=0)[0] * 0)
        # Actually: C[j,k] = max(C[j,k-1], C[j-1,k]) + P[j,k]
        # This is sequential in k, so we can't fully vectorize the k-loop
        # But we can do it with a simple loop over m which is typically small
        c = np.empty(m)
        c[0] = max(prev[0], 0.0) + P[j, 0]
        for k in range(1, m):
            c[k] = max(c[k-1], prev[k]) + P[j, k]
        C[j] = c
    return C[-1, -1]

def _makespan_fast(processing_times, perm):
    """Fast makespan computation using list-based approach for small m."""
    n = len(perm)
    m = processing_times.shape[1]
    if n == 0 or m == 0:
        return 0.0
    P = processing_times
    # Use list of floats for speed
    prev_row = [0.0] * m
    curr_row = [0.0] * m
    for j in range(n):
        job = perm[j]
        if j == 0:
            s = 0.0
            for k in range(m):
                s += P[job, k]
                curr_row[k] = s
        else:
            c0 = prev_row[0]
            for k in range(m):
                val = P[job, k]
                if k == 0:
                    curr_row[0] = c0 + val
                else:
                    cm1 = curr_row[k-1]
                    pk = prev_row[k]
                    curr_row[k] = (cm1 if cm1 > pk else pk) + val
        prev_row, curr_row = curr_row, prev_row
    return prev_row[-1]

def _insertion_cost(processing_times, perm, job, pos):
    """Compute makespan of inserting `job` at position `pos` in `perm`."""
    n = len(perm)
    m = processing_times.shape[1]
    if m == 0:
        return 0.0
    P = processing_times
    # Build the new permutation implicitly and compute makespan
    # C[j][k] = max(C[j][k-1], C[j-1][k]) + P[perm[j]][k]
    prev_row = [0.0] * m
    curr_row = [0.0] * m
    for j in range(n + 1):
        if j < pos:
            job_j = perm[j]
        else:
            job_j = job if j == pos else perm[j - 1]
        if j == 0:
            s = 0.0
            for k in range(m):
                s += P[job_j, k]
                curr_row[k] = s
        else:
            c0 = prev_row[0]
            for k in range(m):
                val = P[job_j, k]
                if k == 0:
                    curr_row[0] = c0 + val
                else:
                    cm1 = curr_row[k-1]
                    pk = prev_row[k]
                    curr_row[k] = (cm1 if cm1 > pk else pk) + val
        prev_row, curr_row = curr_row, prev_row
    return prev_row[-1]

def _neh_standard(processing_times, rng=None):
    """Standard NEH: sort by total processing time descending, insert at best position."""
    n, m = processing_times.shape
    if n == 0:
        return []
    totals = processing_times.sum(axis=1)
    if rng is not None:
        noise = rng.uniform(0, 1e-6, size=n)
        order = np.argsort(-totals + noise)
    else:
        order = np.argsort(-totals)
    perm = []
    for job in order:
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(perm) + 1):
            ms = _insertion_cost(processing_times, perm, job, pos)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    return perm

def _neh_first_last(processing_times, rng=None):
    """NEH variant: first job is the one with max total, then insert best."""
    n, m = processing_times.shape
    if n == 0:
        return []
    totals = processing_times.sum(axis=1)
    if rng is not None:
        noise = rng.uniform(0, 1e-6, size=n)
        totals_noisy = totals + noise
    else:
        totals_noisy = totals
    # Start with the job having max total
    first_job = int(np.argmax(totals_noisy))
    perm = [first_job]
    remaining = [j for j in range(n) if j != first_job]
    # Sort remaining by total descending
    if rng is not None:
        noise = rng.uniform(0, 1e-6, size=len(remaining))
        rem_order = sorted(remaining, key=lambda j: -(totals_noisy[j] + noise[remaining.index(j)]))
    else:
        rem_order = sorted(remaining, key=lambda j: -totals[j])
    for job in rem_order:
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(perm) + 1):
            ms = _insertion_cost(processing_times, perm, job, pos)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    return perm

def _neh_last_machine(processing_times, rng=None):
    """NEH variant: sort by last machine processing time descending."""
    n, m = processing_times.shape
    if n == 0:
        return []
    if m == 0:
        return list(range(n))
    last_col = processing_times[:, m - 1]
    if rng is not None:
        noise = rng.uniform(0, 1e-6, size=n)
        order = np.argsort(-last_col + noise)
    else:
        order = np.argsort(-last_col)
    perm = []
    for job in order:
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(perm) + 1):
            ms = _insertion_cost(processing_times, perm, job, pos)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    return perm

def _random_perm(processing_times, rng):
    """Random permutation."""
    n = processing_times.shape[0]
    return list(rng.permutation(n))

def _tabu_search(processing_times, perm, deadline, rng=None):
    """Tabu search with insertion moves."""
    n = len(perm)
    if n < 2:
        return list(perm), _makespan_fast(processing_times, perm)
    
    if rng is None:
        rng = np.random.default_rng(42)
    
    current_perm = list(perm)
    current_ms = _makespan_fast(processing_times, current_perm)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    tabu_tenure = max(5, n // 2)
    tabu_list = {}  # (job, pos) -> iteration when it becomes non-tabu
    iteration = 0
    
    while time.time() < deadline:
        improved = False
        best_move = None
        best_move_ms = current_ms
        best_move_i = -1
        best_move_j = -1
        best_move_is_tabu = False
        
        for i in range(n):
            if time.time() >= deadline:
                break
            job = current_perm[i]
            remaining = current_perm[:i] + current_perm[i+1:]
            for j in range(len(remaining) + 1):
                if j == i:
                    continue
                # Check if this move is tabu
                move_key = (job, j)
                if move_key in tabu_list and tabu_list[move_key] > iteration:
                    is_tabu = True
                else:
                    is_tabu = False
                
                ms = _insertion_cost(processing_times, remaining, job, j)
                
                # Aspiration: accept tabu move if it improves best
                if is_tabu and ms >= best_ms:
                    continue
                
                if ms < best_move_ms:
                    best_move_ms = ms
                    best_move_i = i
                    best_move_j = j
                    best_move_is_tabu = is_tabu
        
        if best_move_i == -1:
            break
        
        # Make the reverse move tabu
        # The reverse move would be inserting job at position best_move_i in the new perm
        # Actually, we tabu the move (job, best_move_j) for tabu_tenure iterations
        job_moved = current_perm[best_move_i]
        tabu_list[(job_moved, best_move_j)] = iteration + tabu_tenure
        
        # Apply the move
        remaining = current_perm[:best_move_i] + current_perm[best_move_i+1:]
        current_perm = remaining[:best_move_j] + [job_moved] + remaining[best_move_j:]
        current_ms = best_move_ms
        
        if current_ms < best_ms:
            best_ms = current_ms
            best_perm = list(current_perm)
        
        iteration += 1
    
    return best_perm, best_ms

def _or_opt_local_search(processing_times, perm, deadline):
    """Local search with Or-opt moves (relocate blocks of 1-3)."""
    n = len(perm)
    if n < 2:
        return list(perm), _makespan_fast(processing_times, perm)
    
    current_perm = list(perm)
    current_ms = _makespan_fast(processing_times, current_perm)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            for block_len in [1, 2, 3]:
                if i + block_len > n:
                    continue
                block = current_perm[i:i+block_len]
                remaining = current_perm[:i] + current_perm[i+block_len:]
                best_pos = i
                best_ms = current_ms
                for j in range(len(remaining) + 1):
                    # Skip positions equivalent to original
                    if j == i or (j >= i and j - i < block_len and j <= i + block_len - 1):
                        # More carefully: skip if inserting block at j gives same order
                        if j == i:
                            continue
                        if j == i + block_len:
                            continue
                    new_perm = remaining[:j] + block + remaining[j:]
                    ms = _makespan_fast(processing_times, new_perm)
                    if ms < best_ms:
                        best_ms = ms
                        best_pos = j
                if best_ms < current_ms:
                    current_perm = remaining[:best_pos] + block + remaining[best_pos:]
                    current_ms = best_ms
                    improved = True
                    break
            if improved:
                break
    
    return current_perm, current_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Args:
        processing_times: np.ndarray of shape (jobs, machines)
        
    Returns:
        List of job indices in the order they should be processed.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    n, m = processing_times.shape
    if n == 0:
        return []
    if n == 1:
        return [0]
    if m == 0:
        return list(range(n))
    
    rng = np.random.default_rng(12345)
    
    # Generate multiple candidate permutations from different heuristics
    candidates = []
    
    # Standard NEH (deterministic)
    candidates.append(_neh_standard(processing_times))
    
    # First-last NEH (deterministic)
    candidates.append(_neh_first_last(processing_times))
    
    # Last-machine NEH (deterministic)
    candidates.append(_neh_last_machine(processing_times))
    
    # Randomized NEH variants
    for i in range(10):
        if time.time() > deadline:
            break
        candidates.append(_neh_standard(processing_times, rng))
    
    for i in range(10):
        if time.time() > deadline:
            break
        candidates.append(_neh_first_last(processing_times, rng))
    
    for i in range(10):
        if time.time() > deadline:
            break
        candidates.append(_neh_last_machine(processing_times, rng))
    
    # A few random starts
    for i in range(3):
        if time.time() > deadline:
            break
        candidates.append(_random_perm(processing_times, rng))
    
    # Evaluate all candidates and find the best
    best_perm = candidates[0]
    best_ms = _makespan_fast(processing_times, best_perm)
    
    for perm in candidates[1:]:
        if time.time() > deadline:
            break
        ms = _makespan_fast(processing_times, perm)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # Tabu search on the best candidate
    if time.time() < deadline - 5:
        improved_perm, improved_ms = _tabu_search(processing_times, best_perm, deadline, rng)
        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms
    
    # Or-opt local search
    if time.time() < deadline - 1:
        improved_perm, improved_ms = _or_opt_local_search(processing_times, best_perm, deadline)
        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
