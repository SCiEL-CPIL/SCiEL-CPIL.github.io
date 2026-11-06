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

def _compute_prefix_completions(perm, processing_times):
    """
    Compute completion time vectors for every prefix of the permutation.
    prefix_comps[i] is the completion vector after processing perm[0..i].
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    prefix_comps = [None] * n_jobs
    if n_jobs == 0:
        return prefix_comps
    
    prev_row = np.zeros(n_machines)
    for i in range(n_jobs):
        job = perm[i]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
        prefix_comps[i] = prev_row.copy()
        
    return prefix_comps

def _compute_suffix_completions(perm, processing_times):
    """
    Compute "backward" completion vectors for every suffix of the permutation.
    suffix_comps[i] represents the vector D such that Makespan(V, perm[i..]) = max_k (V[k] + D[k]).
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return [np.zeros(n_machines)] # Dummy
    
    suffix_comps = [None] * (n_jobs + 1)
    suffix_comps[n_jobs] = np.zeros(n_machines)
    
    for i in range(n_jobs - 1, -1, -1):
        job = perm[i]
        pt = processing_times[job]
        D_next = suffix_comps[i+1]
        
        W = np.full(n_machines, -np.inf)
        W[0] = pt[0]
        
        for m in range(1, n_machines):
            W += pt[m]
            if W[m] < pt[m]:
                W[m] = pt[m]
        
        max_D_next = np.max(suffix_comps[i+1])
        suffix_comps[i] = W + max_D_next
        
    return suffix_comps

def _eval_swap_incremental(perm, i, processing_times, prefix_comps, suffix_comps):
    """
    Evaluate makespan of swapping perm[i] and perm[i+1].
    """
    n_machines = processing_times.shape[1]
    
    if i == 0:
        C_prev = np.zeros(n_machines)
    else:
        C_prev = prefix_comps[i-1]
    
    job_a = perm[i]
    job_b = perm[i+1]
    
    pt_b = processing_times[job_b]
    pt_a = processing_times[job_a]
    
    C_b = np.zeros(n_machines)
    C_b[0] = C_prev[0] + pt_b[0]
    for m in range(1, n_machines):
        C_b[m] = max(C_prev[m], C_b[m-1]) + pt_b[m]
        
    C_a = np.zeros(n_machines)
    C_a[0] = C_b[0] + pt_a[0]
    for m in range(1, n_machines):
        C_a[m] = max(C_b[m], C_a[m-1]) + pt_a[m]
        
    if i + 2 >= len(perm):
        return float(np.max(C_a))
    else:
        D_suffix = suffix_comps[i+2]
        return float(np.max(C_a + D_suffix))

def _eval_insert_incremental(perm, i, j, processing_times, prefix_comps, suffix_comps):
    """
    Evaluate makespan of moving perm[i] to position j.
    """
    n_machines = processing_times.shape[1]
    
    if i == j:
        return _makespan(perm, processing_times)
    
    if i < j:
        if i == 0:
            C_prev = np.zeros(n_machines)
        else:
            C_prev = prefix_comps[i-1]
            
        C_mid = C_prev
        for k in range(i+1, j+1):
            job = perm[k]
            pt = processing_times[job]
            C_new = np.zeros(n_machines)
            C_new[0] = C_mid[0] + pt[0]
            for m in range(1, n_machines):
                C_new[m] = max(C_mid[m], C_new[m-1]) + pt[m]
            C_mid = C_new
            
        job_a = perm[i]
        pt_a = processing_times[job_a]
        C_a = np.zeros(n_machines)
        C_a[0] = C_mid[0] + pt_a[0]
        for m in range(1, n_machines):
            C_a[m] = max(C_mid[m], C_a[m-1]) + pt_a[m]
            
        if j + 1 >= len(perm):
            return float(np.max(C_a))
        else:
            D_suffix = suffix_comps[j+1]
            return float(np.max(C_a + D_suffix))
            
    else:
        if j == 0:
            C_prev = np.zeros(n_machines)
        else:
            C_prev = prefix_comps[j-1]
            
        job_a = perm[i]
        pt_a = processing_times[job_a]
        C_a = np.zeros(n_machines)
        C_a[0] = C_prev[0] + pt_a[0]
        for m in range(1, n_machines):
            C_a[m] = max(C_prev[m], C_a[m-1]) + pt_a[m]
            
        C_mid = C_a
        for k in range(j, i):
            job = perm[k]
            pt = processing_times[job]
            C_new = np.zeros(n_machines)
            C_new[0] = C_mid[0] + pt[0]
            for m in range(1, n_machines):
                C_new[m] = max(C_mid[m], C_new[m-1]) + pt[m]
            C_mid = C_new
            
        if i + 1 >= len(perm):
            return float(np.max(C_mid))
        else:
            D_suffix = suffix_comps[i+1]
            return float(np.max(C_mid + D_suffix))

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

def _neh_order(processing_times, rng=None, sort_key='total'):
    """Generate NEH heuristic ordering with optional randomization and sort key."""
    if rng is None:
        rng = random.Random()
    
    n_jobs, n_machines = processing_times.shape
    if sort_key == 'total':
        scores = processing_times.sum(axis=1)
    elif sort_key == 'max':
        scores = processing_times.max(axis=1)
    elif sort_key == 'var':
        scores = processing_times.std(axis=1)
    else:
        scores = processing_times.sum(axis=1)
        
    order = list(np.argsort(-scores))
    
    # Add slight randomization to break ties and diversify
    if n_jobs > 2:
        for _ in range(min(3, n_jobs // 5)):
            i = rng.randint(0, n_jobs - 2)
            if rng.random() < 0.3:
                order[i], order[i+1] = order[i+1], order[i]
    
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        partial, _ = _insert_best_position(job, partial, processing_times)
    
    return partial

def _swap_adjacent(perm, i):
    """Create new permutation with adjacent swap at position i."""
    new_perm = list(perm)
    new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
    return new_perm

def _move_job(perm, job, new_pos):
    """Move job to new position."""
    perm = [j for j in perm if j != job]
    return perm[:new_pos] + [job] + perm[new_pos:]

def _local_search_2opt(perm, processing_times, max_iterations=500, deadline=None):
    """
    Local search with 2-opt (adjacent swap) and insert moves using incremental evaluation.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    
    for it in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        prefix_comps = _compute_prefix_completions(best_perm, processing_times)
        suffix_comps = _compute_suffix_completions(best_perm, processing_times)
        
        improved = False
        
        # Try adjacent swaps
        for i in range(n_jobs - 1):
            ms = _eval_swap_incremental(best_perm, i, processing_times, prefix_comps, suffix_comps)
            if ms < best_ms - 1e-10:
                best_perm[i], best_perm[i+1] = best_perm[i+1], best_perm[i]
                best_ms = ms
                improved = True
                break
        
        if improved:
            continue
        
        # If no swap improvement, try insert moves
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
            job = best_perm[i]
            for pos in range(n_jobs):
                if pos == i or pos == i + 1:
                    continue
                ms = _eval_insert_incremental(best_perm, i, pos, processing_times, prefix_comps, suffix_comps)
                if ms < best_ms - 1e-10:
                    new_perm = _move_job(best_perm, job, pos)
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_perm

def _simulated_annealing(perm, processing_times, start_time, deadline, rng=None):
    """
    Simple simulated annealing with adjacent swap and insert moves.
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
    
    initial_temp = current_ms * 0.1
    cooling_rate = 0.999
    
    temp = initial_temp
    
    while time.time() < deadline:
        if temp < 1e-10:
            break
        
        move_type = rng.random()
        if move_type < 0.5:
            i = rng.randint(0, n_jobs - 2)
            new_perm = _swap_adjacent(current_perm, i)
        else:
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if j == i or j == i + 1:
                continue
            job = current_perm[i]
            new_perm = _move_job(current_perm, job, j)
        
        new_ms = _makespan(new_perm, processing_times)
        delta = new_ms - current_ms
        
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_perm = new_perm
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_perm = list(current_perm)
                best_ms = current_ms
        
        temp *= cooling_rate
    
    return best_perm

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin: 5s before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    candidates = []
    
    # 1. LPT
    lpt_perm = list(np.argsort(-processing_times.sum(axis=1)))
    candidates.append(lpt_perm)
    
    # 2. NEH (Total)
    if time.time() < deadline:
        neh_perm = _neh_order(processing_times, rng, sort_key='total')
        candidates.append(neh_perm)
    
    # 3. NEH (Max)
    if time.time() < deadline:
        neh_max_perm = _neh_order(processing_times, random.Random(123), sort_key='max')
        candidates.append(neh_max_perm)
        
    # 4. NEH (Var)
    if time.time() < deadline:
        neh_var_perm = _neh_order(processing_times, random.Random(456), sort_key='var')
        candidates.append(neh_var_perm)

    # Evaluate all candidates and keep the best
    best_perm = None
    best_ms = float('inf')
    for perm in candidates:
        ms = _makespan(perm, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = list(perm)
    
    # Local search polish on best solution
    if time.time() < deadline and n_jobs > 1:
        remaining = deadline - time.time()
        if n_jobs <= 30:
            max_iter = 2000
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
    
    # Simulated annealing if time remains
    if time.time() < deadline - 10.0 and n_jobs > 1:
        sa_deadline = deadline - 10.0
        sa_perm = _simulated_annealing(best_perm, processing_times, start_time, sa_deadline, rng)
        sa_ms = _makespan(sa_perm, processing_times)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
        
        # Final local search polish
        if time.time() < deadline - 5.0:
            max_iter_final = 500 if n_jobs <= 50 else 200
            final_polish = _local_search_2opt(best_perm, processing_times, max_iterations=max_iter_final, deadline=deadline - 5.0)
            final_ms = _makespan(final_polish, processing_times)
            if final_ms < best_ms:
                best_perm = final_polish
                best_ms = final_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
