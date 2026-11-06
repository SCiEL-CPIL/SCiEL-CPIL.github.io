"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _solve_dp(pt):
    """
    Solve flow shop optimally using DP over subsets for small N (<= 12).
    Returns (makespan, permutation) or (None, None) if N is too large or timeout.
    """
    n_jobs = len(pt)
    n_machines = len(pt[0])
    
    if n_jobs > 12:
        return None, None
        
    size = 1 << n_jobs
    dp_vec = [None] * size
    dp_pred = [None] * size
    
    # Initialize single job states
    for j in range(n_jobs):
        mask = 1 << j
        dp_vec[mask] = [None] * n_jobs
        dp_pred[mask] = [None] * n_jobs
        dp_vec[mask][j] = list(pt[j])
        dp_pred[mask][j] = -1
        
    # Iterate masks
    for mask in range(1, size):
        if dp_vec[mask] is None:
            continue
        
        for last_job in range(n_jobs):
            if not (mask & (1 << last_job)):
                continue
            if dp_vec[mask][last_job] is None:
                continue
            
            prev_comp = dp_vec[mask][last_job]
            
            # Try adding next_job not in mask
            for next_job in range(n_jobs):
                if not (mask & (1 << next_job)):
                    new_mask = mask | (1 << next_job)
                    if dp_vec[new_mask] is None:
                        dp_vec[new_mask] = [None] * n_jobs
                        dp_pred[new_mask] = [None] * n_jobs
                    
                    # Calculate new completion vector
                    new_comp = [0.0] * n_machines
                    new_comp[0] = prev_comp[0] + pt[next_job][0]
                    
                    for m in range(1, n_machines):
                        val = prev_comp[m] if prev_comp[m] > new_comp[m-1] else new_comp[m-1]
                        new_comp[m] = val + pt[next_job][m]
                    
                    # Update
                    if dp_vec[new_mask][next_job] is None or new_comp[-1] < dp_vec[new_mask][next_job][-1]:
                        dp_vec[new_mask][next_job] = new_comp
                        dp_pred[new_mask][next_job] = last_job

    # Find best full mask
    full_mask = size - 1
    best_makespan = float('inf')
    best_last_job = -1
    
    if dp_vec[full_mask] is None:
        return None, None
        
    for last_job in range(n_jobs):
        if dp_vec[full_mask][last_job] is not None:
            makespan = dp_vec[full_mask][last_job][-1]
            if makespan < best_makespan:
                best_makespan = makespan
                best_last_job = last_job

    if best_last_job == -1:
        return None, None

    # Reconstruct path
    mask = full_mask
    last_job = best_last_job
    path = []
    while last_job != -1:
        path.append(last_job)
        prev_job = dp_pred[mask][last_job]
        mask = mask ^ (1 << last_job)
        last_job = prev_job
        
    path.reverse()
    return best_makespan, path

def _calc_makespan(pt, perm):
    """
    Calculate makespan for a given permutation using optimized Python lists.
    pt: list of lists of processing times
    perm: list of job indices
    """
    n_machines = len(pt[0])
    completion = [0.0] * n_machines
    
    for job in perm:
        row = pt[job]
        # Machine 0
        completion[0] += row[0]
        # Machines 1..M-1
        # C[i, m] = max(C[i-1, m], C[i, m-1]) + P[i, m]
        # completion[m] currently holds C[i-1, m]
        # completion[m-1] currently holds C[i, m-1] (updated in this loop)
        for m in range(1, n_machines):
            # max(C[i-1, m], C[i, m-1])
            if completion[m] < completion[m-1]:
                completion[m] = completion[m-1]
            completion[m] += row[m]
    
    return completion[-1]

def _eval_insertion(pt, perm, i, j):
    """
    Evaluate makespan of sequence formed by removing job at i and inserting at j.
    i: index of job to remove
    j: index where job is inserted (in the new sequence indices, effectively shifting)
    If i < j, we remove i, and insert at j-1 (because indices shift left).
    If i > j, we remove i, and insert at j.
    """
    n_jobs = len(perm)
    n_machines = len(pt[0])
    
    if i == j:
        # No change
        return _calc_makespan(pt, perm)
        
    job_to_move = perm[i]
    row = pt[job_to_move]
    
    # Create a copy of the permutation without job i
    # To avoid O(N) copy for every evaluation, we can simulate.
    # But for simplicity and correctness, let's use a list slice or manual simulation.
    # Manual simulation is faster.
    
    # We need to compute the makespan of the new sequence.
    # New sequence: perm[0:i] + perm[i+1:j+1] + [job_to_move] + perm[j+1:] (if i<j)
    # Or: perm[0:j] + [job_to_move] + perm[j:i] + perm[i+1:] (if i>j)
    
    # Instead of building the list, we simulate the flow.
    # This is O(N*M) per evaluation.
    
    completion = [0.0] * n_machines
    
    # Process prefix before i (if i>0) or before j (if i<j, prefix is 0..i-1)
    # Case 1: i < j
    # Sequence: 0..i-1, i+1..j, i, j+1..N-1
    # Case 2: i > j
    # Sequence: 0..j-1, i, j..i-1, i+1..N-1
    
    if i < j:
        # 1. Process 0..i-1
        for k in range(i):
            r = pt[perm[k]]
            completion[0] += r[0]
            for m in range(1, n_machines):
                if completion[m] < completion[m-1]:
                    completion[m] = completion[m-1]
                completion[m] += r[m]
        
        # 2. Process i+1..j-1 (these shift left)
        for k in range(i + 1, j):
            r = pt[perm[k]]
            completion[0] += r[0]
            for m in range(1, n_machines):
                if completion[m] < completion[m-1]:
                    completion[m] = completion[m-1]
                completion[m] += r[m]
        
        # 3. Process job i
        completion[0] += row[0]
        for m in range(1, n_machines):
            if completion[m] < completion[m-1]:
                completion[m] = completion[m-1]
            completion[m] += row[m]
        
        # 4. Process j..N-1
        for k in range(j, n_jobs):
            r = pt[perm[k]]
            completion[0] += r[0]
            for m in range(1, n_machines):
                if completion[m] < completion[m-1]:
                    completion[m] = completion[m-1]
                completion[m] += r[m]
                
    else: # i > j
        # 1. Process 0..j-1
        for k in range(j):
            r = pt[perm[k]]
            completion[0] += r[0]
            for m in range(1, n_machines):
                if completion[m] < completion[m-1]:
                    completion[m] = completion[m-1]
                completion[m] += r[m]
        
        # 2. Process job i
        completion[0] += row[0]
        for m in range(1, n_machines):
            if completion[m] < completion[m-1]:
                completion[m] = completion[m-1]
            completion[m] += row[m]
        
        # 3. Process j..i-1
        for k in range(j, i):
            r = pt[perm[k]]
            completion[0] += r[0]
            for m in range(1, n_machines):
                if completion[m] < completion[m-1]:
                    completion[m] = completion[m-1]
                completion[m] += r[m]
        
        # 4. Process i+1..N-1
        for k in range(i + 1, n_jobs):
            r = pt[perm[k]]
            completion[0] += r[0]
            for m in range(1, n_machines):
                if completion[m] < completion[m-1]:
                    completion[m] = completion[m-1]
                completion[m] += r[m]
                
    return completion[-1]

def _neh_heuristic(pt):
    """
    NEH heuristic: 
    1. Sort jobs by total processing time descending.
    2. Insert each job into the best position in the current sequence.
    """
    n_jobs = len(pt)
    
    # Sort by total processing time descending
    totals = [sum(row) for row in pt]
    order = sorted(range(n_jobs), key=lambda x: totals[x], reverse=True)
    
    # Start with the first job
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting at each position
        for pos in range(len(sequence) + 1):
            # Construct new sequence
            # Optimization: Instead of slicing, we can evaluate directly?
            # For NEH, N is small enough that slicing is fine.
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            ms = _calc_makespan(pt, new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence

def _local_search(pt, sequence, deadline):
    """
    Best-Improvement Local Search with time limit.
    Uses insertion moves.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(pt, current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_new_makespan = current_makespan
        best_move_i = -1
        best_move_j = -1
        
        # Try all pairs of positions (i, j)
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            for j in range(n_jobs):
                if j == i:
                    continue
                
                new_makespan = _eval_insertion(pt, current_seq, i, j)
                
                if new_makespan < best_new_makespan:
                    best_new_makespan = new_makespan
                    best_move_i = i
                    best_move_j = j
        
        if best_move_i != -1 and best_new_makespan < current_makespan:
            # Perform the best move
            job_to_move = current_seq.pop(best_move_i)
            if best_move_j > best_move_i:
                insert_idx = best_move_j - 1
            else:
                insert_idx = best_move_j
            current_seq.insert(insert_idx, job_to_move)
            current_makespan = best_new_makespan
            improved = True
        else:
            # No improvement found in this pass
            # Optional: Accept a non-improving move (SA) or break
            # To avoid getting stuck, we can try a random perturbation if time allows
            # But for now, break to save time for potential restarts?
            # No, just break the local search loop.
            pass
            
    return current_seq, current_makespan

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Convert to list of lists for faster Python loop access
    pt = [list(row) for row in processing_times]
    
    n_jobs = len(pt)
    
    # Step 0: Try DP for small instances
    if n_jobs <= 12:
        dp_makespan, dp_perm = _solve_dp(pt)
        if dp_perm is not None:
            return dp_perm
            
    # Step 1: NEH heuristic
    initial_seq = _neh_heuristic(pt)
    
    # Step 2: Local search with time limit
    best_seq, best_makespan = _local_search(pt, initial_seq, deadline)
    
    # Step 3: If time remains, try a few random restarts or perturbations?
    # Given the time limit, and the cost of NEH, we might not have time for many restarts.
    # But if N is small, we can.
    # Let's check remaining time.
    remaining = deadline - time.time()
    if remaining > 10.0:
        # Try a V-shaped heuristic initial solution
        # V-shaped: SPT for first half, LPT for second half
        totals = [sum(row) for row in pt]
        spt_order = sorted(range(n_jobs), key=lambda x: totals[x])
        lpt_order = sorted(range(n_jobs), key=lambda x: totals[x], reverse=True)
        
        # Simple V-shape: First N/2 SPT, Last N/2 LPT?
        # Actually, standard V-shape is SPT then LPT.
        # Let's just use NEH again but with random tie-breaking?
        # Or just return the best found.
        pass

    return best_seq


# EVOLVE-BLOCK-END
