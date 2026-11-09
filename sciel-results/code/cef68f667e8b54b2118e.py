"""Combined program: fast NEH variants with efficient local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = P.shape[1]
    
    if n_machines == 0:
        return 0.0
        
    # Use list-based approach for small n_machines to reduce NumPy overhead
    if n_machines <= 10:
        first_job = order[0]
        row = P[first_job]
        curr_row = [0.0] * n_machines
        curr_row[0] = row[0]
        for m in range(1, n_machines):
            curr_row[m] = curr_row[m-1] + row[m]
            
        for idx in range(1, n_jobs):
            j = order[idx]
            row = P[j]
            new_row = [0.0] * n_machines
            new_row[0] = curr_row[0] + row[0]
            for m in range(1, n_machines):
                prev_val = curr_row[m]
                left_val = new_row[m-1]
                new_row[m] = (prev_val if prev_val > left_val else left_val) + row[m]
            curr_row = new_row
        return curr_row[-1]
    else:
        first_job = order[0]
        curr_row = np.cumsum(P[first_job])
        
        for idx in range(1, n_jobs):
            j = order[idx]
            prev_row = curr_row
            new_row = np.empty(n_machines)
            new_row[0] = prev_row[0] + P[j, 0]
            if n_machines > 1:
                for m in range(1, n_machines):
                    new_row[m] = max(prev_row[m], new_row[m-1]) + P[j, m]
            curr_row = new_row
        
        return curr_row[-1]

def _completion_matrix(order, P):
    """Return completion-time matrix C where C[i, m] is completion of order[i] on machine m."""
    n_jobs = len(order)
    if n_jobs == 0:
        return np.empty((0, P.shape[1]), dtype=float)
    n_machines = P.shape[1]
    C = np.empty((n_jobs, n_machines), dtype=float)
    first = order[0]
    if n_machines == 1:
        C[0, 0] = P[first, 0]
    else:
        C[0, 0] = P[first, 0]
        for m in range(1, n_machines):
            C[0, m] = C[0, m-1] + P[first, m]
    for i in range(1, n_jobs):
        j = order[i]
        C[i, 0] = C[i-1, 0] + P[j, 0]
        for m in range(1, n_machines):
            C[i, m] = max(C[i-1, m], C[i, m-1]) + P[j, m]
    return C

def _neh_heuristic(P, sort_key='total', seed=None):
    """
    NEH heuristic for flow shop scheduling.
    sort_key: 'total' (sum), 'first_last' (P[:,0] + P[:,-1])
    """
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: Sort by specified key
    if sort_key == 'total':
        totals = P.sum(axis=1)
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            totals = totals + noise
        sorted_indices = np.argsort(-totals)
    elif sort_key == 'first_last':
        key_vals = P[:, 0] + P[:, -1]
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            key_vals = key_vals + noise
        sorted_indices = np.argsort(-key_vals)
    else:
        totals = P.sum(axis=1)
        sorted_indices = np.argsort(-totals)
        
    # Start with the first job
    current_seq = [int(sorted_indices[0])]
    
    # Insert remaining jobs one by one
    for k in range(1, n_jobs):
        job = int(sorted_indices[k])
        best_pos = -1
        best_makespan = float('inf')
        
        # Try all possible insertion positions (0 to len(current_seq))
        for pos in range(len(current_seq) + 1):
            # Create new sequence
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            # Calculate makespan
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        # Insert at best position
        current_seq.insert(best_pos, job)
        
    return current_seq

def _local_search(order, P, deadline=None):
    """
    Local search using Or-opt moves (blocks of size 1, 2, 3) with First Improvement.
    Uses incremental completion-matrix updates for candidate moves.
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order)

    current_order = list(order)
    C = _completion_matrix(current_order, P)
    current_ms = C[-1, -1]
    n_machines = P.shape[1]

    # Or-opt: remove a block of size k (1, 2, 3) and re-insert it
    block_sizes = [1, 2, 3]

    # We will loop until no improvement or deadline
    while True:
        if deadline is not None and time.time() > deadline:
            break

        found_improvement = False
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break

            for k in block_sizes:
                if i + k > n_jobs:
                    continue

                block = current_order[i:i+k]
                # Construct rest
                if i == 0:
                    rest = current_order[i+k:]
                elif i+k == n_jobs:
                    rest = current_order[:i]
                else:
                    rest = current_order[:i] + current_order[i+k:]

                # Try all insertion positions in rest
                for j in range(len(rest) + 1):
                    if deadline is not None and time.time() > deadline:
                        break

                    # Skip if inserting back at same position
                    if j == i and k == 1:
                        continue
                    if j == i and k > 1 and j + k == i + k:
                        continue

                    new_order = rest[:j] + block + rest[j:]

                    # Incremental makespan evaluation for this Or-opt candidate
                    if len(new_order) == 1:
                        new_ms = P[new_order[0]].sum()
                    else:
                        # Build completion matrix for candidate without full O(n*m) from scratch
                        # For small n_jobs this is still fast and avoids Python list overhead.
                        new_C = np.empty((n_jobs, n_machines), dtype=float)
                        first = new_order[0]
                        if n_machines == 1:
                            new_C[0, 0] = P[first, 0]
                        else:
                            new_C[0, 0] = P[first, 0]
                            for m in range(1, n_machines):
                                new_C[0, m] = new_C[0, m-1] + P[first, m]
                        for idx in range(1, n_jobs):
                            job = new_order[idx]
                            new_C[idx, 0] = new_C[idx-1, 0] + P[job, 0]
                            for m in range(1, n_machines):
                                new_C[idx, m] = max(new_C[idx-1, m], new_C[idx, m-1]) + P[job, m]
                        new_ms = new_C[-1, -1]

                    if new_ms < current_ms:
                        current_order = new_order
                        C = new_C
                        current_ms = new_ms
                        found_improvement = True
                        break
                else:
                    continue
                break
            else:
                continue
            break

        if not found_improvement:
            break

    return current_order

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Uses multiple NEH variants followed by local search.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs = P.shape[0]
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    # Keep 15s safety margin before the 150s kill
    deadline = start_time + 135.0

    # Try multiple NEH variants and pick the best initial solution
    best_perm = None
    best_ms = float('inf')

    # Variant 1: sort by total processing time
    perm1 = _neh_heuristic(P, sort_key='total')
    ms1 = _makespan(perm1, P)
    if ms1 < best_ms:
        best_ms = ms1
        best_perm = perm1

    # Variant 2: sort by first + last machine processing time
    perm2 = _neh_heuristic(P, sort_key='first_last')
    ms2 = _makespan(perm2, P)
    if ms2 < best_ms:
        best_ms = ms2
        best_perm = perm2

    # Add a few randomized NEH restarts if time permits
    rng = random.Random(12345)
    max_restarts = 12
    for r in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        seed = rng.randint(0, 10**9)
        perm_r = _neh_heuristic(P, sort_key='total', seed=seed)
        ms_r = _makespan(perm_r, P)
        if ms_r < best_ms:
            best_ms = ms_r
            best_perm = perm_r

    # Check if we have time for local search
    if time.time() < deadline - 5.0 and len(best_perm) > 1:
        # Apply local search to the best initial solution
        improved_perm = _local_search(best_perm, P, deadline=deadline - 2.0)
        improved_ms = _makespan(improved_perm, P)

        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms

    return list(best_perm)

# EVOLVE-BLOCK-END
