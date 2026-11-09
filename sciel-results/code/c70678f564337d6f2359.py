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
    # completion[j][k] = completion time of job perm[j] on machine k
    # We track the completion time on the last machine for each job
    # and the availability time of each machine.
    # Actually, for flow shop:
    # C[j][k] = max(C[j][k-1], C[j-1][k]) + p[perm[j]][k]
    # where C[-1][k] = 0 and C[j][-1] = 0
    # We only need the final makespan, but for incremental moves we might need more.
    # For now, simple O(n*m) calculation.
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
    
    # Total processing time for each job
    totals = processing_times.sum(axis=1)
    
    # Sort jobs in non-increasing order of total processing time
    # For ties, use random order if random_state provided, else stable sort
    if random_state is not None:
        # Add small random noise to break ties randomly
        noise = random_state.uniform(0, 1e-6, size=n)
        sort_keys = -totals + noise
    else:
        sort_keys = -totals
    
    sorted_jobs = np.argsort(sort_keys)
    
    # Insert jobs one by one into the best position
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

def _or_opt_improve(processing_times, perm, max_iterations=10000, deadline=None):
    """Improve permutation using Or-opt moves (relocate blocks of 1-3 jobs)."""
    n = len(perm)
    if n < 2:
        return perm
    
    current_perm = list(perm)
    current_ms = _makespan_from_perm(processing_times, current_perm)
    
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        if deadline is not None and time.time() > deadline:
            break
        
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            for block_len in [1, 2, 3]:
                if i + block_len > n:
                    continue
                # Extract block
                block = current_perm[i:i+block_len]
                # Remove block
                remaining = current_perm[:i] + current_perm[i+block_len:]
                
                # Try inserting block at each position in remaining
                for j in range(len(remaining) + 1):
                    if j == i or (j == i + block_len):
                        continue
                    # Skip positions that are equivalent to original
                    # We need to be careful about indices after removal
                    # If j > i, the position in remaining corresponds to original j+block_len
                    # Let's just compute the new permutation and check
                    new_perm = remaining[:j] + block + remaining[j:]
                    new_ms = _makespan_from_perm(processing_times, new_perm)
                    if new_ms < current_ms:
                        current_perm = new_perm
                        current_ms = new_ms
                        improved = True
                        # Restart the loop for this i? Or just continue?
                        # Typically, after an improvement, we restart the neighborhood search
                        # for simplicity, we break out and restart
                        break
                if improved:
                    break
            if improved:
                break
    
    return current_perm

def _swap_improve(processing_times, perm, max_iterations=10000, deadline=None):
    """Improve permutation using swap moves."""
    n = len(perm)
    if n < 2:
        return perm
    
    current_perm = list(perm)
    current_ms = _makespan_from_perm(processing_times, current_perm)
    
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        if deadline is not None and time.time() > deadline:
            break
        
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(i + 1, n):
                # Swap i and j
                new_perm = current_perm[:]
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_ms = _makespan_from_perm(processing_times, new_perm)
                if new_ms < current_ms:
                    current_perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break
            if improved:
                break
    
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
    deadline = start_time + 110  # 10 seconds safety margin
    
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
    
    # Local search: Or-opt
    improved_perm = _or_opt_improve(processing_times, best_perm, max_iterations=5000, deadline=deadline)
    improved_ms = _makespan_from_perm(processing_times, improved_perm)
    if improved_ms < best_ms:
        best_perm = improved_perm
        best_ms = improved_ms
    
    # Local search: Swap
    if time.time() < deadline:
        improved_perm = _swap_improve(processing_times, best_perm, max_iterations=5000, deadline=deadline)
        improved_ms = _makespan_from_perm(processing_times, improved_perm)
        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms
    
    # Another round of Or-opt if time permits
    if time.time() < deadline - 5:
        improved_perm = _or_opt_improve(processing_times, best_perm, max_iterations=3000, deadline=deadline)
        improved_ms = _makespan_from_perm(processing_times, improved_perm)
        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
