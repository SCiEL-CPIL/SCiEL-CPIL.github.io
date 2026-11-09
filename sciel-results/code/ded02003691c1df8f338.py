# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan_vec(processing_times, perm):
    """Compute makespan using vectorized NumPy."""
    n = len(perm)
    m = processing_times.shape[1]
    if n == 0 or m == 0:
        return 0.0
    P = processing_times[perm]  # (n, m)
    C = np.zeros((n, m))
    C[0] = np.cumsum(P[0])
    for j in range(1, n):
        prev = C[j-1]
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
    first_job = int(np.argmax(totals_noisy))
    perm = [first_job]
    remaining = [j for j in range(n) if j != first_job]
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

def _cds_heuristic(processing_times, rng=None):
    """CDS heuristic: build sequence by inserting jobs in reverse sort order (from last to first)."""
    n, m = processing_times.shape
    if n == 0:
        return []
    if n == 1:
        return [0]
    totals = processing_times.sum(axis=1)
    if rng is not None:
        noise = rng.uniform(0, 1e-6, size=n)
        order = np.argsort(-totals + noise)
    else:
        order = np.argsort(-totals)
    # CDS: start from the last job in the sorted order, insert earlier jobs at best position
    perm = [int(order[-1])]
    for k in range(n - 2, -1, -1):
        job = int(order[k])
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
            remaining = current_perm[:i] + current_perm[i+1:]
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
                break
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
    avg_processing = np.mean(processing_times)
    T0 = max(current_ms * 0.05, avg_processing * n * 0.1, 1.0)
    T_end = T0 * 1e-6
    cooling_rate = 0.995
    T = T0
    steps_since_best = 0
    max_stall = 500
    while time.time() < deadline and T > T_end:
        if rng.random() < 0.6:
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
            job = current_perm[i]
            remaining = current_perm[:i] + current_perm[i+1:]
            if j > i:
                j -= 1
            new_perm = remaining[:j] + [job] + remaining[j:]
        else:
            i = rng.integers(0, n - 1)
            j = rng.integers(i + 1, n)
            new_perm = current_perm[:i] + current_perm[i:j+1][::-1] + current_perm[j+1:]
        new_ms = _makespan_fast(processing_times, new_perm)
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_perm = list(current_perm)
                best_ms = current_ms
                steps_since_best = 0
            else:
                steps_since_best += 1
        T *= cooling_rate
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
                    if j == i or j == i + block_len:
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

def _pso_diversify(processing_times, best_perm, best_ms, deadline, rng):
    """Discrete PSO-style diversification: perturb best with random swaps, re-optimize via insertion."""
    n = len(best_perm)
    if n < 2:
        return list(best_perm), best_ms
    current_best = list(best_perm)
    current_best_ms = best_ms
    n_particles = min(8, max(3, n // 2))
    for p in range(n_particles):
        if time.time() > deadline:
            break
        # Create a perturbed particle: apply 2-4 random swaps to best
        particle = list(current_best)
        n_swaps = rng.integers(2, min(5, n - 1) + 1)
        for _ in range(n_swaps):
            i, j = rng.choice(n, size=2, replace=False)
            particle[i], particle[j] = particle[j], particle[i]
        # Re-optimize the perturbed particle via local search (insertion)
        opt_perm, opt_ms = _local_search(processing_times, particle, deadline, rng)
        if opt_ms < current_best_ms:
            current_best = opt_perm
            current_best_ms = opt_ms
    return current_best, current_best_ms

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

    # CDS heuristic (deterministic) - structurally different insertion order
    candidates.append(_cds_heuristic(processing_times))

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

    # Randomized CDS variants
    for i in range(5):
        if time.time() > deadline:
            break
        candidates.append(_cds_heuristic(processing_times, rng))

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

    # PSO-style diversification: perturb best and re-optimize
    if time.time() < deadline - 1:
        improved_perm, improved_ms = _pso_diversify(processing_times, best_perm, best_ms, deadline, rng)
        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms

    # Final high-precision polish: run local search + or-opt again with remaining time
    if time.time() < deadline - 0.5:
        improved_perm, improved_ms = _local_search(processing_times, best_perm, deadline)
        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms
    if time.time() < deadline - 0.1:
        improved_perm, improved_ms = _or_opt_local_search(processing_times, best_perm, deadline)
        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms

    return list(best_perm)
# EVOLVE-BLOCK-END
