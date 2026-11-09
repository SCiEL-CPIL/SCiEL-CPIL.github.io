"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan_from_perm(processing_times, perm):
    """Compute makespan for a given permutation."""
    n = len(perm)
    m = processing_times.shape[1]
    if n == 0 or m == 0:
        return 0.0
    completion = np.zeros((n, m))
    for j in range(n):
        job = perm[j]
        for k in range(m):
            if j > 0 and k > 0:
                completion[j, k] = max(completion[j, k-1], completion[j-1, k])
            elif j > 0:
                completion[j, k] = completion[j-1, k]
            elif k > 0:
                completion[j, k] = completion[j, k-1]
            else:
                completion[j, k] = 0.0
            completion[j, k] += processing_times[job, k]
    return completion[-1, -1]

def _neh(processing_times, random_state=None):
    """NEH heuristic for flow shop scheduling."""
    n, m = processing_times.shape
    if n == 0:
        return []
    
    totals = processing_times.sum(axis=1)
    
    if random_state is not None:
        noise = random_state.uniform(0, 1e-6, size=n)
        sort_keys = -totals + noise
    else:
        sort_keys = -totals
    
    sorted_jobs = np.argsort(sort_keys)
    
    perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_from_perm(processing_times, new_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm

def _vns_search(processing_times, perm, deadline, max_iterations=200):
    """Variable Neighborhood Search with systematic neighborhood cycling."""
    n = len(perm)
    if n < 2:
        return list(perm)
    
    current_perm = list(perm)
    current_ms = _makespan_from_perm(processing_times, current_perm)
    
    # Neighborhoods of increasing size
    # N1: adjacent swaps
    # N2: 2-opt block reversals (reverse a segment)
    # N3: 3-opt segment relocations (move a block of 1-3 jobs)
    
    def _try_adjacent_swap(p, ms):
        """Try all adjacent swaps, return best improvement."""
        n = len(p)
        best_p = None
        best_ms = ms
        for i in range(n - 1):
            new_p = p[:]
            new_p[i], new_p[i+1] = new_p[i+1], new_p[i]
            new_ms = _makespan_from_perm(processing_times, new_p)
            if new_ms < best_ms:
                best_ms = new_ms
                best_p = new_p
        return best_p, best_ms
    
    def _try_2opt_reversal(p, ms):
        """Try all 2-opt reversals (reverse a segment), return best improvement."""
        n = len(p)
        best_p = None
        best_ms = ms
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_p = p[:]
                new_p[i:j+1] = reversed(new_p[i:j+1])
                new_ms = _makespan_from_perm(processing_times, new_p)
                if new_ms < best_ms:
                    best_ms = new_ms
                    best_p = new_p
        return best_p, best_ms
    
    def _try_3opt_relocation(p, ms):
        """Try 3-opt segment relocations (move blocks of 1-3 jobs), return best improvement."""
        n = len(p)
        best_p = None
        best_ms = ms
        for block_len in [1, 2, 3]:
            for i in range(n - block_len + 1):
                block = p[i:i+block_len]
                remaining = p[:i] + p[i+block_len:]
                for j in range(len(remaining) + 1):
                    if j == i or j == i + block_len:
                        continue
                    new_p = remaining[:j] + block + remaining[j:]
                    new_ms = _makespan_from_perm(processing_times, new_p)
                    if new_ms < best_ms:
                        best_ms = new_ms
                        best_p = new_p
        return best_p, best_ms
    
    neighborhoods = [
        ("swap", _try_adjacent_swap),
        ("reverse", _try_2opt_reversal),
        ("relocate", _try_3opt_relocation),
    ]
    
    iter_count = 0
    while iter_count < max_iterations:
        if time.time() > deadline:
            break
        iter_count += 1
        
        improved = False
        for name, func in neighborhoods:
            if time.time() > deadline:
                break
            new_perm, new_ms = func(current_perm, current_ms)
            if new_perm is not None and new_ms < current_ms:
                current_perm = new_perm
                current_ms = new_ms
                improved = True
                break  # Restart from smallest neighborhood after improvement
        
        if not improved:
            # Perturbation: random swap of two random positions
            if n >= 2:
                i, j = np.random.default_rng(iter_count).choice(n, 2, replace=False)
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                current_ms = _makespan_from_perm(processing_times, current_perm)
    
    return current_perm

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Args:
        processing_times: np.ndarray of shape (jobs, machines)
        
    Returns:
        List of job indices in the order they should be processed.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10 seconds safety margin
    
    n, m = processing_times.shape
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # Start with NEH
    best_perm = _neh(processing_times)
    best_ms = _makespan_from_perm(processing_times, best_perm)
    
    # Try a few randomized NEH restarts
    rng = np.random.default_rng(42)
    for _ in range(5):
        if time.time() > deadline:
            break
        alt_perm = _neh(processing_times, random_state=rng)
        alt_ms = _makespan_from_perm(processing_times, alt_perm)
        if alt_ms < best_ms:
            best_perm = alt_perm
            best_ms = alt_ms
    
    # VNS search with systematic neighborhood cycling
    if time.time() < deadline - 5:
        vns_result = _vns_search(
            processing_times, best_perm, deadline=deadline, max_iterations=200
        )
        vns_ms = _makespan_from_perm(processing_times, vns_result)
        if vns_ms < best_ms:
            best_perm = vns_result
            best_ms = vns_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
