# EVOLVE-BLOCK-START
import numpy as np
import time

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

def _local_search(processing_times, perm, deadline, rng=None):
    """Local search with insertion moves (move a job to a different position)."""
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
            job = current_perm[i]
            # Remove job from position i
            remaining = current_perm[:i] + current_perm[i+1:]
            # Try inserting at each position in remaining
            best_pos = i
            best_ms = current_ms
            for j in range(len(remaining) + 1):
                if j == i:
                    continue
                ms = _insertion_cost(processing_times, remaining, job, j)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = j
            if best_ms < current_ms:
                current_perm = remaining[:best_pos] + [job] + remaining[best_pos:]
                current_ms = best_ms
                improved = True
                break  # Restart scan after improvement
    
    return current_perm, current_ms

def _simulated_annealing(processing_times, perm, deadline, rng=None):
    """Simulated Annealing with 2-opt block-reversal and insertion moves."""
    n = len(perm)
    if n < 2:
        return list(perm), _makespan_fast(processing_times, perm)
    
    if rng is None:
        rng = np.random.default_rng(42)
    
    current_perm = list(perm)
    current_ms = _makespan_fast(processing_times, current_perm)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # Initial temperature based on typical makespan scale
    avg_processing = np.mean(processing_times)
    T0 = max(current_ms * 0.05, avg_processing * n * 0.1, 1.0)
    T_end = T0 * 1e-6
    cooling_rate = 0.995
    
    T = T0
    steps_since_best = 0
    max_stall = 500
    
    while time.time() < deadline and T > T_end:
        # Choose move type: 60% insertion, 40% 2-opt block reversal
        if rng.random() < 0.6:
            # Insertion move
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
            job = current_perm[i]
            remaining = current_perm[:i] + current_perm[i+1:]
            # Adjust j if needed
            if j > i:
                j -= 1
            new_perm = remaining[:j] + [job] + remaining[j:]
        else:
            # 2-opt block reversal
            i = rng.integers(0, n - 1)
            j = rng.integers(i + 1, n)
            new_perm = current_perm[:i] + current_perm[i:j+1][::-1] + current_perm[j+1:]
        
        new_ms = _makespan_fast(processing_times, new_perm)
        delta = new_ms - current_ms
        
        # Acceptance criterion
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_perm = list(current_perm)
                best_ms = current_ms
                steps_since_best = 0
            else:
                steps_since_best += 1
        
        # Cool down
        T *= cooling_rate
        
        # If stuck for too long, reheat slightly
        if steps_since_best > max_stall:
            T = max(T, T0 * 0.01)
            steps_since_best = 0
    
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
    
    # Local search on the best candidate
    if time.time() < deadline - 2:
        improved_perm, improved_ms = _local_search(processing_times, best_perm, deadline)
        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms
    
    # Simulated Annealing to escape local optima
    if time.time() < deadline - 1:
        improved_perm, improved_ms = _simulated_annealing(processing_times, best_perm, deadline, rng)
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
