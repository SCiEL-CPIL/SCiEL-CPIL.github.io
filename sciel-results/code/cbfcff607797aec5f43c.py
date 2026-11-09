"""Improved program for flow-shop permutation construction using diverse NEH variants and local search."""
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
        # Pre-extract rows to avoid P[j, m] indexing overhead in inner loop?
        # P is a numpy array. P[j] is a view.
        # Let's just use lists for curr_row.
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
                # max(curr_row[m], new_row[m-1]) + row[m]
                prev_val = curr_row[m]
                left_val = new_row[m-1]
                new_row[m] = (prev_val if prev_val > left_val else left_val) + row[m]
            curr_row = new_row
        return curr_row[-1]
    else:
        # Use NumPy for larger M
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

def _neh_heuristic(P, sort_key='total', deadline=None, seed=None):
    """
    NEH heuristic for flow shop scheduling.
    sort_key: 'total' (sum), 'first_last' (P[:,0] + P[:,-1]), 'random'
    """
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: Sort by specified key
    if sort_key == 'total':
        totals = P.sum(axis=1)
        # Add small random noise to break ties differently in randomized runs
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            totals = totals + noise
        sorted_indices = np.argsort(-totals)
    elif sort_key == 'first_last':
        # Heuristic: jobs with high processing on first and last machines are critical
        key_vals = P[:, 0] + P[:, -1]
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            key_vals = key_vals + noise
        sorted_indices = np.argsort(-key_vals)
    elif sort_key == 'random':
        if seed is not None:
            rng = np.random.RandomState(seed)
            sorted_indices = rng.permutation(n_jobs)
        else:
            sorted_indices = np.random.permutation(n_jobs)
    else:
        totals = P.sum(axis=1)
        sorted_indices = np.argsort(-totals)
        
    # Start with the first job
    current_seq = [int(sorted_indices[0])]
    
    # Insert remaining jobs one by one
    for k in range(1, n_jobs):
        # Check deadline
        if deadline is not None:
            if time.time() > deadline:
                break
        
        job = int(sorted_indices[k])
        best_pos = -1
        best_makespan = float('inf')
        
        # Try all possible insertion positions (0 to len(current_seq))
        for pos in range(len(current_seq) + 1):
            # Check deadline inside inner loop for large n_jobs
            if deadline is not None and pos % 10 == 0:
                if time.time() > deadline:
                    break
            
            # Create new sequence
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            # Calculate makespan
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        # If deadline was hit inside inner loop, best_pos might be -1 or partial
        if best_pos == -1:
            # If we couldn't find a valid position due to timeout, just append to end
            current_seq.append(job)
        else:
            # Insert at best position
            current_seq.insert(best_pos, job)
        
    return current_seq

def _makespan_from_order(order, P, C=None):
    """
    Compute makespan for a given job order. If C (completion matrix) is provided,
    use it to incrementally update rather than recompute from scratch.
    Returns (makespan, completion_matrix).
    """
    n_jobs = len(order)
    n_machines = P.shape[1]
    
    if n_jobs == 0:
        return 0.0, np.zeros((0, n_machines))
    
    if C is None:
        C = np.zeros((n_jobs, n_machines))
    
    # First job
    j0 = order[0]
    C[0, 0] = P[j0, 0]
    for m in range(1, n_machines):
        C[0, m] = C[0, m-1] + P[j0, m]
    
    # Subsequent jobs
    for idx in range(1, n_jobs):
        j = order[idx]
        C[idx, 0] = C[idx-1, 0] + P[j, 0]
        for m in range(1, n_machines):
            C[idx, m] = max(C[idx-1, m], C[idx, m-1]) + P[j, m]
    
    return C[n_jobs-1, n_machines-1], C

def _local_search(order, P, deadline=None):
    """
    Local search using Or-opt moves with incremental makespan updates.
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return order
    
    n_machines = P.shape[1]
    current_order = list(order)
    current_ms, C = _makespan_from_order(current_order, P)
    
    # Or-opt: remove a block of size k and re-insert it
    block_sizes = [1, 2, 3]
    
    while True:
        if deadline is not None and time.time() > deadline:
            break
            
        improved = False
        
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
                
            for k in block_sizes:
                if i + k > n_jobs:
                    continue
                
                # Extract the block
                block = current_order[i:i+k]
                
                # Construct the rest of the sequence
                if i == 0:
                    rest = current_order[i+k:]
                elif i+k == n_jobs:
                    rest = current_order[:i]
                else:
                    rest = current_order[:i] + current_order[i+k:]
                
                rest_len = len(rest)  # n_jobs - k
                
                for j in range(rest_len + 1):
                    if j == i:
                        continue
                    
                    # Build new order
                    if j == 0:
                        new_order = block + rest
                    elif j == rest_len:
                        new_order = rest + block
                    else:
                        new_order = rest[:j] + block + rest[j:]
                    
                    # Compute makespan for new order
                    new_ms, _ = _makespan_from_order(new_order, P)
                    
                    if new_ms < current_ms:
                        current_order = new_order
                        current_ms = new_ms
                        C, _ = _makespan_from_order(current_order, P)
                        improved = True
                        break
                else:
                    continue
                break
            else:
                continue
            break
            
        if not improved:
            break

    return current_order

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Uses diverse NEH heuristics followed by local search improvement.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs = P.shape[0]
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Set a deadline with a safety margin
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Generate multiple candidates using different NEH variants
    candidates = []
    
    # 1. Standard NEH
    seq1 = _neh_heuristic(P, sort_key='total', deadline=deadline)
    candidates.append(seq1)
    
    # 2. NEH with first/last machine focus
    if time.time() < deadline - 10.0: # Ensure we have time for another construction
        seq2 = _neh_heuristic(P, sort_key='first_last', deadline=deadline)
        candidates.append(seq2)
        
    # 3. Reverse NEH (cheap diversification)
    if seq1:
        seq_rev = list(reversed(seq1))
        candidates.append(seq_rev)

    # 4. Randomized NEH (if n_jobs is small enough or time permits)
    if time.time() < deadline - 20.0 and n_jobs <= 200:
        # Use a fixed seed for reproducibility in this run, or random?
        # The task doesn't specify reproducibility, but deterministic is safer for debugging.
        # However, random restarts can help escape bad local minima.
        # Let's use a few random restarts if time is plentiful.
        for seed in [42, 123, 999]:
            if time.time() > deadline - 20.0:
                break
            seq_r = _neh_heuristic(P, sort_key='random', deadline=deadline, seed=seed)
            candidates.append(seq_r)

    # Find the best candidate
    best_seq = candidates[0]
    best_ms = _makespan(best_seq, P)
    
    for seq in candidates[1:]:
        if len(seq) < n_jobs:
            # Complete if necessary
            remaining = [j for j in range(n_jobs) if j not in seq]
            seq.extend(remaining)
        ms = _makespan(seq, P)
        if ms < best_ms:
            best_ms = ms
            best_seq = seq

    # Ensure best_seq is a valid permutation
    if len(best_seq) < n_jobs:
        remaining = [j for j in range(n_jobs) if j not in best_seq]
        best_seq.extend(remaining)
        
    # Apply local search if there is time
    if time.time() < deadline - 5.0:
        ls_deadline = deadline - 2.0
        best_seq = _local_search(best_seq, P, deadline=ls_deadline)
    
    return list(best_seq)

# EVOLVE-BLOCK-END
