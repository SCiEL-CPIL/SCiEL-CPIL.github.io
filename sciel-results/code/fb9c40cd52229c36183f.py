"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, processing_times):
    """Calculate the makespan for a given job order in a flow shop."""
    if not order:
        return 0.0
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    
    curr = np.zeros(n_machines)
    for j in order:
        curr[0] += processing_times[j, 0]
        if n_machines > 1:
            for m in range(1, n_machines):
                curr[m] = max(curr[m-1], curr[m]) + processing_times[j, m]
    return curr[-1]


def _compute_completion_times(order, processing_times):
    """Compute completion times C[j][m] for each job position j and machine m."""
    n = len(order)
    n_machines = processing_times.shape[1]
    if n == 0 or n_machines == 0:
        return np.zeros((n, n_machines))
    
    C = np.zeros((n, n_machines))
    for j in range(n):
        job = order[j]
        C[j, 0] = processing_times[job, 0]
        if n_machines > 1:
            for m in range(1, n_machines):
                C[j, m] = max(C[j-1, m], C[j, m-1]) + processing_times[job, m]
    return C


def _delta_swap(i, order, C, processing_times):
    """
    Compute the change in makespan when swapping adjacent jobs at positions i and i+1.
    Returns the new makespan if the swap is applied, or None if it's worse.
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    if i >= n - 1:
        return C[-1, -1]
    
    j1 = order[i]
    j2 = order[i+1]
    
    # The swap only affects positions i and i+1 in the completion time matrix
    # We need to recompute C[i][*] and C[i+1][*]
    
    # New C[i][0]
    new_C_i = np.zeros(n_machines)
    new_C_i[0] = processing_times[j2, 0]
    
    # New C[i+1][0]
    new_C_i1_0 = new_C_i[0] + processing_times[j1, 0]
    
    if n_machines > 1:
        for m in range(1, n_machines):
            # C[i][m] = max(C[i-1][m], C[i][m-1]) + p[j2][m]
            prev_cm = C[i-1, m] if i > 0 else 0.0
            new_C_i[m] = max(prev_cm, new_C_i[m-1]) + processing_times[j2, m]
        
        # C[i+1][m] = max(C[i][m], C[i+1][m-1]) + p[j1][m]
        # Note: C[i+1][m-1] uses the NEW C[i+1][m-1] which we compute iteratively
        # But we also need the old C[i+1][m] for positions after i+1 to remain valid
    
    # The makespan is the max of:
    # 1. The completion time at position i+1, last machine (new)
    # 2. The completion time at position i+2 (or later), last machine (unchanged, but depends on C[i+1])
    
    # Actually, the key insight: after swapping, C[i+1][m] changes for all m.
    # This affects C[i+2][m] for all m, which cascades.
    # So we can't just look at local positions - we need to recompute from i+1 onwards.
    
    # For efficiency, let's compute the new makespan by simulating from position i onwards
    # But that's still O((n-i)*M).
    
    # Alternative: compute new C[i+1][last] and see if it's worse than old makespan
    # If new C[i+1][last] > old makespan, the swap is definitely worse.
    # If new C[i+1][last] < old makespan, we need to check if downstream is affected.
    
    # Let's compute the new completion times from position i onwards
    new_C_i1 = np.zeros(n_machines)
    new_C_i1[0] = new_C_i1_0
    if n_machines > 1:
        for m in range(1, n_machines):
            new_C_i1[m] = max(new_C_i[m], new_C_i1[m-1]) + processing_times[j1, m]
    
    # Now check if the makespan increases
    old_makespan = C[-1, -1]
    
    # The new makespan is determined by:
    # - new_C_i1[-1] if i+1 is the last position
    # - or the cascade effect on subsequent positions
    
    # If i+1 is the last position:
    if i + 1 == n - 1:
        return new_C_i1[-1]
    
    # Otherwise, we need to check if the change propagates
    # The downstream positions j > i+1 have C[j][m] = max(C[j-1][m], C[j][m-1]) + p[j][m]
    # Since C[i+1][m] changed, C[i+2][m] may change, etc.
    
    # Quick check: if new_C_i1[m] <= C[i+1][m] for all m, then downstream is unaffected
    # (because C[j][m] is non-decreasing in j for fixed m, and the max operation is monotone)
    
    # Actually, C[j][m] depends on C[j-1][m] and C[j][m-1]. If C[i+1][m] decreases,
    # then C[i+2][m] might decrease, which might decrease C[i+2][m+1], etc.
    # This is complex. Let's just recompute from i+1 to end if needed.
    
    # For now, let's do a full recomputation from position i to end to be safe
    # But this is O((n-i)*M), which for i near the start is O(n*M)
    
    # Better approach: just recompute the full makespan for the swapped sequence
    # But that's what we're trying to avoid.
    
    # Let's use a different strategy: compute the new makespan incrementally
    # Start from position i, use the new C[i], C[i+1], then continue with the same formula
    
    new_C = C.copy()
    new_C[i] = new_C_i
    new_C[i+1] = new_C_i1
    
    for j in range(i+2, n):
        job = order[j]
        new_C[j, 0] = new_C[j-1, 0] + processing_times[job, 0]
        if n_machines > 1:
            for m in range(1, n_machines):
                new_C[j, m] = max(new_C[j-1, m], new_C[j, m-1]) + processing_times[job, m]
    
    return new_C[-1, -1]


def _neh_heuristic(processing_times):
    """Generate initial permutation using NEH heuristic."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    job_sums = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-job_sums))
    
    current_sequence = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current_sequence) + 1):
            new_seq = current_sequence[:pos] + [job] + current_sequence[pos:]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current_sequence.insert(best_pos, job)
    
    return current_sequence


def _adjacent_swap_ls(order, processing_times, deadline):
    """Local search using adjacent swaps with incremental makespan updates."""
    n = len(order)
    if n <= 1:
        return order, _makespan(order, processing_times)
    
    current_order = list(order)
    C = _compute_completion_times(current_order, processing_times)
    current_makespan = C[-1, -1]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            
            new_makespan = _delta_swap(i, current_order, C, processing_times)
            
            if new_makespan < current_makespan:
                # Apply the swap
                current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                current_makespan = new_makespan
                C = _compute_completion_times(current_order, processing_times)
                improved = True
                break
    
    return current_order, current_makespan


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Phase 1: NEH Heuristic
    current_order = _neh_heuristic(processing_times)
    current_makespan = _makespan(current_order, processing_times)
    
    best_order = current_order[:]
    best_makespan = current_makespan
    
    # Phase 2: Adjacent Swap Local Search
    ls_deadline = deadline - 1.0
    if time.time() < ls_deadline:
        current_order, current_makespan = _adjacent_swap_ls(current_order, processing_times, ls_deadline)
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_order = current_order[:]
    
    # Phase 3: If time remains, try random restarts
    remaining_time = deadline - time.time()
    if remaining_time > 1.0 and n_jobs > 5:
        import random
        rng = random.Random(42)
        
        max_restarts = min(5, int(remaining_time / 2.0))
        
        for _ in range(max_restarts):
            if time.time() > deadline - 0.5:
                break
            
            random_order = list(range(n_jobs))
            rng.shuffle(random_order)
            
            ls_deadline_restart = time.time() + max(0.1, remaining_time / 2.0)
            rand_order, rand_ms = _adjacent_swap_ls(random_order, processing_times, ls_deadline_restart)
            
            if rand_ms < best_makespan:
                best_makespan = rand_ms
                best_order = rand_order[:]
    
    return best_order

# EVOLVE-BLOCK-END
