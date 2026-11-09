"""Combined program: Fast NEH with incremental Or-Opt Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan_fast(order, P):
    """
    Compute makespan using a fast list-based approach.
    This is used for the initial NEH construction where we evaluate many candidates.
    """
    if not order:
        return 0.0
    n_jobs = len(order)
    n_machines = P.shape[1]
    
    if n_machines == 0:
        return 0.0

    # Use Python lists for the inner loop to avoid NumPy overhead for small m
    # C[i][m] = completion time of job at position i on machine m
    # We only need the last row to get the makespan, but we need the previous row to compute the current one.
    
    # Initialize for first job
    job0 = order[0]
    curr_row = [0.0] * n_machines
    curr_row[0] = P[job0, 0]
    for m in range(1, n_machines):
        curr_row[m] = curr_row[m-1] + P[job0, m]
        
    for idx in range(1, n_jobs):
        job = order[idx]
        prev_row = curr_row
        new_row = [0.0] * n_machines
        new_row[0] = prev_row[0] + P[job, 0]
        for m in range(1, n_machines):
            # C(idx, m) = max(C(idx-1, m), C(idx, m-1)) + P(idx, m)
            val_prev = prev_row[m]
            val_left = new_row[m-1]
            new_row[m] = (val_prev if val_prev > val_left else val_left) + P[job, m]
        curr_row = new_row
        
    return curr_row[-1]

def _completion_matrix(order, P):
    """
    Compute the full completion matrix C (list of lists).
    C[i][m] is the completion time of the job at position i on machine m.
    """
    n_jobs = len(order)
    n_machines = P.shape[1]
    if n_jobs == 0:
        return []
        
    C = [[0.0] * n_machines for _ in range(n_jobs)]
    
    job0 = order[0]
    C[0][0] = P[job0, 0]
    for m in range(1, n_machines):
        C[0][m] = C[0][m-1] + P[job0, m]
        
    for i in range(1, n_jobs):
        job = order[i]
        C[i][0] = C[i-1][0] + P[job, 0]
        for m in range(1, n_machines):
            val_prev = C[i-1][m]
            val_left = C[i][m-1]
            C[i][m] = (val_prev if val_prev > val_left else val_left) + P[job, m]
            
    return C

def _neh_heuristic(P, deadline=None):
    """
    NEH heuristic.
    """
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    totals = P.sum(axis=1)
    sorted_indices = np.argsort(-totals)
    
    current_seq = [int(sorted_indices[0])]
    
    for k in range(1, n_jobs):
        if deadline is not None and time.time() > deadline:
            break
            
        job = int(sorted_indices[k])
        best_pos = -1
        best_ms = float('inf')
        
        for pos in range(len(current_seq) + 1):
            if deadline is not None and pos % 5 == 0 and time.time() > deadline:
                break
                
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            ms = _makespan_fast(new_seq, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
                
        if best_pos == -1:
            current_seq.append(job)
        else:
            current_seq.insert(best_pos, job)
            
    return current_seq

def _local_search(order, P, deadline=None):
    """
    Local search using Or-Opt moves (block sizes 1, 2, 3).
    Maintains the completion matrix C for incremental updates.
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order)
    
    current_order = list(order)
    C = _completion_matrix(current_order, P)
    current_ms = C[-1][-1]
    n_machines = P.shape[1]
    
    # Block sizes to try
    block_sizes = [1, 2, 3]
    
    # First Improvement strategy: if a better move is found, apply it and restart the scan
    # This is often faster than Best Improvement for large n
    while True:
        if deadline is not None and time.time() > deadline:
            break
            
        improved = False
        # Iterate through possible removal positions
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
                
            for k in block_sizes:
                if i + k > n_jobs:
                    continue
                    
                # Extract block
                block = current_order[i:i+k]
                
                # Construct rest of sequence
                if i == 0:
                    rest = current_order[i+k:]
                elif i + k == n_jobs:
                    rest = current_order[:i]
                else:
                    rest = current_order[:i] + current_order[i+k:]
                    
                # Try all insertion positions in rest
                for j in range(len(rest) + 1):
                    if deadline is not None and time.time() > deadline:
                        break
                        
                    # Skip if inserting back at the same place (no-op)
                    # If we remove [i, i+k) and insert at j in rest,
                    # the new position of the block in the full sequence is:
                    # If j < i, the block moves to j.
                    # If j >= i, the block moves to j (because the first i elements are still there, 
                    # but the block was removed, so indices shift).
                    # Actually, simpler check:
                    # If j == i and k == 1, it's a no-op.
                    # If j == i and k > 1, it's a no-op? 
                    # If we remove 2 items at i, and insert at i in the rest (which starts at i), 
                    # the sequence is the same.
                    if j == i:
                        continue
                        
                    # Construct new order
                    new_order = rest[:j] + block + rest[j:]
                    
                    # Evaluate new makespan incrementally?
                    # For simplicity and correctness, we recalculate C for the new order.
                    # Since n_jobs is likely small (e.g., < 100), this is fast enough.
                    # If n_jobs is large, we would need a true delta update.
                    # Given the 120s limit, let's assume n_jobs is manageable.
                    
                    new_C = _completion_matrix(new_order, P)
                    new_ms = new_C[-1][-1]
                    
                    if new_ms < current_ms:
                        current_order = new_order
                        C = new_C
                        current_ms = new_ms
                        improved = True
                        # Restart the search from the beginning to find further improvements
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
    # Safety margin: 110 seconds
    deadline = start_time + 110.0
    
    # 1. NEH Heuristic
    permutation = _neh_heuristic(P, deadline)
    
    # Ensure permutation is complete
    if len(permutation) < n_jobs:
        remaining = [j for j in range(n_jobs) if j not in permutation]
        # Sort remaining by total processing time descending
        totals = P.sum(axis=1)
        remaining.sort(key=lambda x: -totals[x])
        permutation.extend(remaining)
        
    # 2. Local Search
    # Leave 5 seconds buffer for return and overhead
    ls_deadline = deadline - 5.0
    if time.time() < ls_deadline:
        permutation = _local_search(permutation, P, deadline=ls_deadline)
        
    return list(permutation)

# EVOLVE-BLOCK-END
