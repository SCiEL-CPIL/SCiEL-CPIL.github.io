"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


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
        cur_row = np.zeros(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
    return float(prev_row[-1])


def _compute_completion_vector(perm, processing_times):
    """Compute the completion time vector for all machines for a permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return np.zeros(n_machines)
    prev_row = np.zeros(n_machines)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
    return prev_row


def _eval_swap_incremental(perm, processing_times, i):
    """
    Evaluate makespan after swapping adjacent jobs at positions i and i+1.
    Uses incremental computation: only recompute from position i onward.
    Returns the new makespan.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    # Compute completion vector up to position i (exclusive)
    prev_row = np.zeros(n_machines)
    for j in range(i):
        job = perm[j]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
    
    # Process swapped jobs at positions i and i+1
    job_a = perm[i+1]  # now at position i
    job_b = perm[i]    # now at position i+1
    
    pt_a = processing_times[job_a]
    pt_b = processing_times[job_b]
    
    cur_row_a = np.zeros(n_machines)
    cur_row_a[0] = prev_row[0] + pt_a[0]
    if n_machines > 1:
        for m in range(1, n_machines):
            cur_row_a[m] = max(prev_row[m], cur_row_a[m-1]) + pt_a[m]
    
    cur_row_b = np.zeros(n_machines)
    cur_row_b[0] = cur_row_a[0] + pt_b[0]
    if n_machines > 1:
        for m in range(1, n_machines):
            cur_row_b[m] = max(cur_row_a[m], cur_row_b[m-1]) + pt_b[m]
    
    # Continue from position i+2 onward
    prev_row = cur_row_b
    for j in range(i+2, n_jobs):
        job = perm[j]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
    
    return float(prev_row[-1])


def _eval_insert_incremental(perm, processing_times, job_idx, new_pos):
    """
    Evaluate makespan after moving job at index job_idx to position new_pos.
    If new_pos > job_idx, the job is moved right; otherwise left.
    Returns the new makespan.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    job = perm[job_idx]
    
    # Build the permutation without the job
    remaining = perm[:job_idx] + perm[job_idx+1:]
    
    # Determine actual insertion position in the remaining list
    if new_pos > job_idx:
        actual_pos = new_pos - 1  # because we removed one element before new_pos
    else:
        actual_pos = new_pos
    
    # Clamp
    actual_pos = max(0, min(actual_pos, len(remaining)))
    
    # Compute completion up to actual_pos
    prev_row = np.zeros(n_machines)
    for j in range(actual_pos):
        jjob = remaining[j]
        pt = processing_times[jjob]
        cur_row = np.zeros(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
    
    # Insert the job
    pt = processing_times[job]
    cur_row = np.zeros(n_machines)
    cur_row[0] = prev_row[0] + pt[0]
    if n_machines > 1:
        for m in range(1, n_machines):
            cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
    prev_row = cur_row
    
    # Continue with rest of remaining
    for j in range(actual_pos, len(remaining)):
        jjob = remaining[j]
        pt = processing_times[jjob]
        cur_row = np.zeros(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
    
    return float(prev_row[-1])


def _insert_best_position(job, partial, processing_times):
    """Insert job into partial list at the position minimizing makespan."""
    n_partial = len(partial)
    if n_partial == 0:
        return [job], _makespan([job], processing_times)
    
    best_pos = 0
    best_ms = float('inf')
    
    for pos in range(n_partial + 1):
        candidate = partial[:pos] + [job] + partial[pos:]
        ms = _makespan(candidate, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_pos = pos
    
    return partial[:best_pos] + [job] + partial[best_pos:], best_ms


def _neh_order(processing_times, rng=None):
    """Generate NEH heuristic ordering with optional randomization."""
    if rng is None:
        rng = random.Random()
    
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    
    # Sort by total processing time descending
    order = list(np.argsort(-totals))
    
    # Add slight randomization to break ties and diversify
    if n_jobs > 2:
        # Swap a few adjacent elements randomly
        for _ in range(min(3, n_jobs // 5)):
            i = rng.randint(0, n_jobs - 2)
            if rng.random() < 0.3:
                order[i], order[i+1] = order[i+1], order[i]
    
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        partial, _ = _insert_best_position(job, partial, processing_times)
    
    return partial


def _neh_order_forward_backward(processing_times, rng=None):
    """NEH with forward and backward passes."""
    if rng is None:
        rng = random.Random()
    
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Forward pass
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        partial, _ = _insert_best_position(job, partial, processing_times)
    
    # Backward pass: remove each job and reinsert at best position
    result = list(partial)
    for i in range(n_jobs - 1, 0, -1):
        # Remove job at position i
        job = result[i]
        remaining = result[:i] + result[i+1:]
        result, _ = _insert_best_position(job, remaining, processing_times)
    
    return result


def _swap_adjacent(perm, i):
    """Create new permutation with adjacent swap at position i."""
    new_perm = list(perm)
    new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
    return new_perm


def _insert_job(perm, job, pos):
    """Insert job at position pos in perm (removing it from current position if present)."""
    if job in perm:
        perm = [j for j in perm if j != job]
    return perm[:pos] + [job] + perm[pos:]


def _move_job(perm, job, new_pos):
    """Move job to new position."""
    perm = [j for j in perm if j != job]
    return perm[:new_pos] + [job] + perm[new_pos:]


def _local_search_2opt(perm, processing_times, max_iterations=200, deadline=None):
    """
    Local search with 2-opt (adjacent swap) and insert moves.
    Uses First-Improvement for both swaps and inserts to maximize iterations within time budget.
    Uses incremental makespan evaluation for speed.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    
    for it in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # 1. Adjacent Swaps (First Improvement) - using incremental eval
        for i in range(n_jobs - 1):
            ms = _eval_swap_incremental(best_perm, processing_times, i)
            if ms < best_ms - 1e-10:
                best_perm = _swap_adjacent(best_perm, i)
                best_ms = ms
                improved = True
                break
        
        # 2. Insert Moves (First Improvement) - using incremental eval
        found_improvement = False
        for i in range(n_jobs):
            job = best_perm[i]
            # Try a limited set of target positions for speed
            candidate_positions = set()
            if n_jobs <= 20:
                candidate_positions = set(range(n_jobs))
            else:
                for offset in [-3, -2, -1, 1, 2, 3]:
                    p = i + offset
                    if 0 <= p < n_jobs:
                        candidate_positions.add(p)
                candidate_positions.add(0)
                candidate_positions.add(n_jobs - 1)
                candidate_positions.add(n_jobs // 2)
            
            for pos in sorted(candidate_positions):
                if pos == i or pos == i + 1:
                    continue
                ms = _eval_insert_incremental(best_perm, processing_times, i, pos)
                if ms < best_ms - 1e-10:
                    best_perm = _move_job(best_perm, job, pos)
                    best_ms = ms
                    improved = True
                    found_improvement = True
                    break
            if found_improvement:
                break
        
        if not improved:
            break
    
    return best_perm


def _simulated_annealing(perm, processing_times, start_time, deadline, rng=None):
    """
    Simulated annealing with adjacent swap and insert moves.
    Uses incremental makespan evaluation for speed.
    """
    if rng is None:
        rng = random.Random()
    
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    current_perm = list(perm)
    current_ms = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # Set initial temperature based on problem size
    initial_temp = current_ms * 0.1
    cooling_rate = 0.9995
    
    temp = initial_temp
    
    while time.time() < deadline:
        if temp < 1e-10:
            break
        
        # Generate neighbor
        move_type = rng.random()
        if move_type < 0.6:
            # Adjacent swap - fast incremental eval
            i = rng.randint(0, n_jobs - 2)
            new_ms = _eval_swap_incremental(current_perm, processing_times, i)
            delta = new_ms - current_ms
            
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                current_perm = _swap_adjacent(current_perm, i)
                current_ms = new_ms
                
                if current_ms < best_ms:
                    best_perm = list(current_perm)
                    best_ms = current_ms
        else:
            # Insert move - incremental eval
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if j == i or j == i + 1:
                continue
            new_ms = _eval_insert_incremental(current_perm, processing_times, i, j)
            delta = new_ms - current_ms
            
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                job = current_perm[i]
                current_perm = _move_job(current_perm, job, j)
                current_ms = new_ms
                
                if current_ms < best_ms:
                    best_perm = list(current_perm)
                    best_ms = current_ms
        
        temp *= cooling_rate
    
    return best_perm


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin: 10s before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    # Generate multiple diverse starting solutions
    candidates = []
    
    # 1. Simple LPT
    lpt_perm = list(np.argsort(-processing_times.sum(axis=1)))
    candidates.append(lpt_perm)
    
    # 2. NEH
    if time.time() < deadline:
        neh_perm = _neh_order(processing_times, rng)
        candidates.append(neh_perm)
    
    # 3. NEH with randomization (diversified)
    if time.time() < deadline:
        neh2_perm = _neh_order(processing_times, random.Random(rng.randint(0, 1000000)))
        candidates.append(neh2_perm)
    
    # 4. NEH forward-backward
    if time.time() < deadline:
        neh_fb_perm = _neh_order_forward_backward(processing_times, rng)
        candidates.append(neh_fb_perm)
    
    # Evaluate all candidates and keep the best
    best_perm = None
    best_ms = float('inf')
    for perm in candidates:
        ms = _makespan(perm, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = list(perm)
    
    # Quick local search on each candidate to find the best starting point for deeper search
    if time.time() < deadline - 10.0 and n_jobs > 1:
        for idx, perm in enumerate(candidates):
            if time.time() > deadline - 10.0:
                break
            quick_polish = _local_search_2opt(perm, processing_times, max_iterations=100, deadline=deadline - 10.0)
            qp_ms = _makespan(quick_polish, processing_times)
            if qp_ms < best_ms:
                best_ms = qp_ms
                best_perm = list(quick_polish)
    
    # Local search polish on best solution - increased iterations due to faster eval
    if time.time() < deadline and n_jobs > 1:
        remaining = deadline - time.time()
        if n_jobs <= 30:
            max_iter = 1500
        elif n_jobs <= 50:
            max_iter = 1000
        elif n_jobs <= 100:
            max_iter = 500
        else:
            max_iter = 200
        
        polished = _local_search_2opt(best_perm, processing_times, max_iterations=max_iter, deadline=deadline)
        polished_ms = _makespan(polished, processing_times)
        if polished_ms < best_ms:
            best_perm = polished
            best_ms = polished_ms
    
    # Simulated annealing if time remains - more iterations due to faster eval
    if time.time() < deadline - 5.0 and n_jobs > 1:
        sa_deadline = deadline - 5.0
        sa_perm = _simulated_annealing(best_perm, processing_times, start_time, sa_deadline, rng)
        sa_ms = _makespan(sa_perm, processing_times)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
        
        # Final local search polish
        if time.time() < deadline - 2.0:
            final_polish = _local_search_2opt(best_perm, processing_times, max_iterations=300, deadline=deadline - 2.0)
            final_ms = _makespan(final_polish, processing_times)
            if final_ms < best_ms:
                best_perm = final_polish
                best_ms = final_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
