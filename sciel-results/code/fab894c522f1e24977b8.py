"""Flow-shop permutation construction using NEH variants + insertion/Or-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    if len(order) == 0:
        return 0.0
    n_machines = P.shape[1]
    completion = np.zeros(n_machines, dtype=np.float64)
    for j in order:
        prev = 0.0
        for m in range(n_machines):
            prev = max(prev, completion[m]) + P[j, m]
            completion[m] = prev
    return float(completion[-1])


def _makespan_from_order_array(order, P):
    """Vectorized makespan computation for numpy array order."""
    if len(order) == 0:
        return 0.0
    n_machines = P.shape[1]
    completion = np.zeros(n_machines, dtype=np.float64)
    for j in order:
        prev = 0.0
        for m in range(n_machines):
            prev = max(prev, completion[m]) + P[j, m]
            completion[m] = prev
    return float(completion[-1])


def _makespan_delta_insert(order, P, job, pos):
    """Compute makespan when inserting job at position pos in order.
    order is a list of job indices (length n), we insert job at pos.
    This is O(n * m) but we only need it for comparison."""
    n = len(order)
    n_machines = P.shape[1]
    # Build the new order
    new_order = order[:pos] + [job] + order[pos:]
    return _makespan(new_order, P)


def _neh_standard(P):
    """Standard NEH heuristic."""
    n_jobs = P.shape[0]
    totals = P.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    perm = []
    for j in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [j] + perm[pos:]
            ms = _makespan(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, int(j))
    
    return perm


def _neh_first_last(P):
    """NEH variant using first machine processing time for sorting."""
    n_jobs = P.shape[0]
    sort_key = P[:, 0]
    sorted_jobs = np.argsort(-sort_key)
    
    perm = []
    for j in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [j] + perm[pos:]
            ms = _makespan(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, int(j))
    
    return perm


def _neh_last_machine(P):
    """NEH variant using last machine processing time for sorting."""
    n_jobs = P.shape[0]
    n_machines = P.shape[1]
    sort_key = P[:, n_machines - 1]
    sorted_jobs = np.argsort(-sort_key)
    
    perm = []
    for j in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [j] + perm[pos:]
            ms = _makespan(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, int(j))
    
    return perm


def _neh_random(P, rng):
    """NEH variant with random sorting."""
    n_jobs = P.shape[0]
    sorted_jobs = rng.permutation(n_jobs)
    
    perm = []
    for j in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [j] + perm[pos:]
            ms = _makespan(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, int(j))
    
    return perm


def _insertion_search(order, P, max_passes=20, time_limit=None, start_time=None):
    """Insertion local search: try moving each job to every other position."""
    n = len(order)
    if n <= 2:
        return list(order), _makespan(order, P)
    
    best_order = list(order)
    best_ms = _makespan(best_order, P)
    
    for _ in range(max_passes):
        if time_limit is not None and start_time is not None:
            if time.time() > start_time + time_limit:
                break
        
        improved = False
        for i in range(n):
            if time_limit is not None and start_time is not None:
                if time.time() > start_time + time_limit:
                    return best_order, best_ms
            job = best_order[i]
            # Remove job from position i
            remaining = best_order[:i] + best_order[i+1:]
            best_pos = i
            best_candidate_ms = best_ms
            
            for pos in range(len(remaining) + 1):
                if pos == i or pos == i + 1:
                    continue
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = _makespan(candidate, P)
                if ms < best_candidate_ms - 1e-10:
                    best_candidate_ms = ms
                    best_pos = pos
            
            if best_pos != i:
                remaining = best_order[:i] + best_order[i+1:]
                best_order = remaining[:best_pos] + [job] + remaining[best_pos:]
                best_ms = best_candidate_ms
                improved = True
        
        if not improved:
            break
    
    return best_order, best_ms


def _or_opt_search(order, P, max_passes=10, time_limit=None, start_time=None):
    """Or-opt local search: try moving blocks of size 1-3 to other positions."""
    n = len(order)
    if n <= 3:
        return list(order), _makespan(order, P)
    
    best_order = list(order)
    best_ms = _makespan(best_order, P)
    
    for _ in range(max_passes):
        if time_limit is not None and start_time is not None:
            if time.time() > start_time + time_limit:
                break
        
        improved = False
        # Try blocks of size 1, 2, 3
        for block_size in [1, 2, 3]:
            if n < block_size + 1:
                continue
            for i in range(n - block_size + 1):
                if time_limit is not None and start_time is not None:
                    if time.time() > start_time + time_limit:
                        return best_order, best_ms
                block = best_order[i:i+block_size]
                # Remove block
                remaining = best_order[:i] + best_order[i+block_size:]
                # Try inserting at all other positions
                for pos in range(len(remaining) + 1):
                    # Skip positions that would result in same or adjacent placement
                    if pos == i or (block_size == 1 and pos == i + 1) or \
                       (block_size > 1 and pos >= i and pos < i + block_size):
                        continue
                    candidate = remaining[:pos] + block + remaining[pos:]
                    ms = _makespan(candidate, P)
                    if ms < best_ms - 1e-10:
                        best_order = candidate
                        best_ms = ms
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_order, best_ms


def _swap_search(order, P, max_passes=10, time_limit=None, start_time=None):
    """Swap local search: try swapping pairs of jobs."""
    n = len(order)
    if n <= 2:
        return list(order), _makespan(order, P)
    
    best_order = list(order)
    best_ms = _makespan(best_order, P)
    
    for _ in range(max_passes):
        if time_limit is not None and start_time is not None:
            if time.time() > start_time + time_limit:
                break
        
        improved = False
        for i in range(n - 1):
            if time_limit is not None and start_time is not None:
                if time.time() > start_time + time_limit:
                    return best_order, best_ms
            for j in range(i + 1, n):
                candidate = best_order[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = _makespan(candidate, P)
                if ms < best_ms - 1e-10:
                    best_order = candidate
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_order, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    P = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = P.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    rng = np.random.RandomState(42)
    
    # Generate multiple NEH variants
    candidates = []
    candidates.append(_neh_standard(P))
    candidates.append(_neh_first_last(P))
    candidates.append(_neh_last_machine(P))
    
    # Add more random NEH variants for diversity
    for _ in range(4):
        candidates.append(_neh_random(P, rng))
    
    # Evaluate all candidates
    best_order = None
    best_ms = float('inf')
    for cand in candidates:
        ms = _makespan(cand, P)
        if ms < best_ms:
            best_ms = ms
            best_order = cand
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_order
    
    # Phase 1: Insertion search on best candidate
    remaining_time = deadline - (time.time() - start_time) - 2.0
    best_order, best_ms = _insertion_search(
        best_order, P, max_passes=30, time_limit=remaining_time, start_time=start_time
    )
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_order
    
    # Phase 2: Or-opt search (block moves)
    remaining_time = deadline - (time.time() - start_time) - 2.0
    best_order, best_ms = _or_opt_search(
        best_order, P, max_passes=10, time_limit=remaining_time, start_time=start_time
    )
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_order
    
    # Phase 3: Another round of insertion search
    remaining_time = deadline - (time.time() - start_time) - 2.0
    best_order, best_ms = _insertion_search(
        best_order, P, max_passes=20, time_limit=remaining_time, start_time=start_time
    )
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_order
    
    # Phase 4: Final Or-opt + insertion polish
    remaining_time = deadline - (time.time() - start_time) - 1.0
    best_order, best_ms = _or_opt_search(
        best_order, P, max_passes=5, time_limit=remaining_time, start_time=start_time
    )
    
    # Final insertion pass if time allows
    elapsed = time.time() - start_time
    if elapsed < deadline - 3:
        remaining_time = deadline - (time.time() - start_time) - 1.0
        best_order, best_ms = _insertion_search(
            best_order, P, max_passes=10, time_limit=remaining_time, start_time=start_time
        )
    
    return best_order
# EVOLVE-BLOCK-END
