import numpy as np
import time
import random
import math

def _makespan(perm, processing_times):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    prev_row = np.zeros(n_machines)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        cur_row = np.empty(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
    return float(prev_row[-1])

def _compute_completion_matrix(perm, processing_times):
    """Compute the full completion time matrix C."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    C = np.zeros((n_jobs, n_machines))
    if n_jobs == 0:
        return C
    
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        if j == 0:
            C[j, 0] = pt[0]
        else:
            C[j, 0] = C[j-1, 0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                C[j, m] = max(C[j-1, m], C[j, m-1]) + pt[m]
    return C

def _eval_insert(perm, C, i, j, processing_times):
    """
    Evaluate the makespan of moving job at index i to index j.
    i: current index of job
    j: target index (insertion position in the sequence WITHOUT the job at i)
    If j > i, the job is moved right. The effective insertion index in the new perm is j.
    If j < i, the job is moved left. The effective insertion index in the new perm is j.
    
    Returns the new makespan.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if i == j or (i + 1 == j):
        # Moving to adjacent position or same position is a swap or no-op.
        # We handle swaps separately or treat as insert.
        # For simplicity, if i+1 == j, it's a swap.
        # If i == j, no change.
        return float(C[-1, -1])

    job = perm[i]
    pt_job = processing_times[job]
    
    # Determine the range of indices to recompute
    # If moving right (i < j):
    # Old: ... i ... j ...
    # New: ... i+1 ... j-1, job, j ...
    # The jobs between i and j shift.
    # We need to recompute from i to n_jobs-1?
    # Actually, if we move job i to j (i < j):
    # New perm: perm[0:i] + perm[i+1:j+1] + [job] + perm[j+1:]
    # The completion times of perm[0:i-1] are unchanged.
    # We need to recompute from i onwards.
    
    # If moving left (i > j):
    # New perm: perm[0:j] + [job] + perm[j:i] + perm[i+1:]
    # Completion times of perm[0:j-1] are unchanged.
    # We need to recompute from j onwards.
    
    start_idx = min(i, j)
    
    # Get the previous row to start recomputation
    if start_idx == 0:
        prev_row = np.zeros(n_machines)
    else:
        prev_row = C[start_idx - 1].copy()
        
    # Construct the new sequence of jobs for the recomputation part
    # We don't want to modify perm yet, just simulate.
    
    new_jobs = []
    # Jobs before start_idx are not included in new_jobs, we start from start_idx
    
    # Logic to build the list of jobs from start_idx to end, with the move applied
    # Original jobs from start_idx to end: perm[start_idx:]
    # We remove job at i and insert at j (relative to the whole list? No, j is the target index in the NEW list? 
    # Usually "insert at j" means the job takes position j in the new list.
    # Let's assume j is the index in the NEW permutation.
    
    # If i < j:
    # New list: perm[0:i] + perm[i+1:j+1] + [job] + perm[j+1:]
    # The part to recompute starts at i.
    # The jobs in the recompute part are: perm[i+1], ..., perm[j], job, perm[j+1], ...
    
    # If i > j:
    # New list: perm[0:j] + [job] + perm[j:i] + perm[i+1:]
    # The part to recompute starts at j.
    # The jobs in the recompute part are: job, perm[j], ..., perm[i-1], perm[i+1], ...
    
    if i < j:
        # Jobs to process: perm[i+1] ... perm[j], then job, then perm[j+1] ...
        # Indices in original perm: i+1 to j
        for k in range(i + 1, j + 1):
            new_jobs.append(perm[k])
        new_jobs.append(job)
        for k in range(j + 1, n_jobs):
            new_jobs.append(perm[k])
    else: # i > j
        new_jobs.append(job)
        for k in range(j, i):
            new_jobs.append(perm[k])
        for k in range(i + 1, n_jobs):
            new_jobs.append(perm[k])
            
    # Now recompute completion times for these new_jobs
    # prev_row is C[start_idx - 1]
    # The first job in new_jobs will be at index start_idx in the new perm.
    
    cur_row = np.empty(n_machines)
    for idx, job_id in enumerate(new_jobs):
        pt = processing_times[job_id]
        pos = start_idx + idx
        
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
        
        # Early termination check?
        # If this is the last job, we are done.
        if pos == n_jobs - 1:
             return float(prev_row[-1])
             
        # Optimization: If the new completion time is <= the old completion time at this position,
        # and this holds for all subsequent positions?
        # We can't easily check "all subsequent" without computing them.
        # But we can check if the current row is <= C[pos] (old).
        # If so, the rest will be <= C[pos+1] etc.
        # However, we still need the final makespan.
        # So we just continue.
        
    return float(prev_row[-1])

def _neh_order(processing_times, rng=None):
    """Generate NEH heuristic ordering."""
    if rng is None:
        rng = random.Random()
    
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    
    # Sort by total processing time descending
    order = list(np.argsort(-totals))
    
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(partial) + 1):
            candidate = partial[:pos] + [job] + partial[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        partial = partial[:best_pos] + [job] + partial[best_pos:]
        
    return partial

def _local_search(perm, processing_times, deadline=None):
    """
    Best Improvement Local Search with Insertion Moves.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    best_perm = list(perm)
    C = _compute_completion_matrix(best_perm, processing_times)
    best_ms = float(C[-1, -1])
    
    improved = True
    while improved:
        improved = False
        if deadline is not None and time.time() > deadline:
            break
            
        best_i = -1
        best_j = -1
        best_new_ms = best_ms
        
        # Try all insertion moves
        for i in range(n_jobs):
            for j in range(n_jobs):
                if i == j or j == i + 1:
                    continue
                
                new_ms = _eval_insert(best_perm, C, i, j, processing_times)
                
                if new_ms < best_new_ms - 1e-10:
                    best_new_ms = new_ms
                    best_i = i
                    best_j = j
                    best_ms_candidate = new_ms # Keep track of the best move found so far in this iteration
        
        if best_i != -1:
            # Apply the move
            job = best_perm[best_i]
            # Remove job
            del best_perm[best_i]
            # Adjust j if i < j
            if best_i < best_j:
                best_j -= 1
            # Insert job
            best_perm.insert(best_j, job)
            
            # Update C
            C = _compute_completion_matrix(best_perm, processing_times)
            best_ms = best_new_ms
            improved = True
    
    return best_perm

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    # Generate multiple diverse starting solutions
    candidates = []
    
    # 1. LPT
    lpt_perm = list(np.argsort(-processing_times.sum(axis=1)))
    candidates.append(lpt_perm)
    
    # 2. NEH
    if time.time() < deadline:
        neh_perm = _neh_order(processing_times, rng)
        candidates.append(neh_perm)
        
    # 3. Randomized NEH
    if time.time() < deadline:
        rng2 = random.Random(rng.randint(0, 1000000))
        neh2_perm = _neh_order(processing_times, rng2)
        candidates.append(neh2_perm)
        
    # 4. Random
    random_perm = list(range(n_jobs))
    rng.shuffle(random_perm)
    candidates.append(random_perm)
    
    # Evaluate all candidates and keep the best
    best_perm = None
    best_ms = float('inf')
    for perm in candidates:
        ms = _makespan(perm, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = list(perm)
            
    # Local search polish
    if time.time() < deadline and n_jobs > 1:
        polished = _local_search(best_perm, processing_times, deadline=deadline)
        polished_ms = _makespan(polished, processing_times)
        if polished_ms < best_ms:
            best_perm = polished
            best_ms = polished_ms
            
    # Iterated Local Search
    # If time remains, perturb and search again
    while time.time() < deadline - 5.0:
        # Perturb best_perm
        perturbed = list(best_perm)
        # Random swap
        n_swaps = rng.randint(1, 3)
        for _ in range(n_swaps):
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            
        # Local search on perturbed
        searched = _local_search(perturbed, processing_times, deadline=deadline)
        searched_ms = _makespan(searched, processing_times)
        
        if searched_ms < best_ms:
            best_perm = searched
            best_ms = searched_ms
        else:
            # Optionally accept worse solutions with SA logic, but for now just restart
            # To ensure diversity, we might just keep the best
            pass

    return list(best_perm)
