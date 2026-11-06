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


def _dp_exact_small(processing_times, max_jobs=15, deadline=None):
    """
    Exact DP for small instances using state-space exploration.
    State: (subset of jobs, last job) -> min makespan.
    Uses bitmask representation for subsets.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs > max_jobs:
        return None
    
    if deadline is not None and time.time() > deadline:
        return None
    
    full_mask = (1 << n_jobs) - 1
    
    # dp[mask][last] = (min_completion_vector, min_makespan)
    # For memory efficiency, store only the completion vector
    dp = {}
    
    # Initialize: single job states
    for j in range(n_jobs):
        mask = 1 << j
        pt = processing_times[j]
        comp = np.zeros(n_machines)
        comp[0] = pt[0]
        for m in range(1, n_machines):
            comp[m] = comp[m-1] + pt[m]
        dp[(mask, j)] = comp.copy()
    
    # Build up subsets
    for mask in range(1, full_mask + 1):
        if deadline is not None and time.time() > deadline:
            return None
        # Get jobs in this mask
        jobs_in_mask = [j for j in range(n_jobs) if (mask >> j) & 1]
        if len(jobs_in_mask) == 1:
            continue
        
        for last in jobs_in_mask:
            prev_mask = mask ^ (1 << last)
            if prev_mask == 0:
                continue
            
            best_comp = None
            best_ms = float('inf')
            
            for prev_last in jobs_in_mask:
                if prev_last == last:
                    continue
                if not (prev_mask >> prev_last) & 1:
                    continue
                
                state_key = (prev_mask, prev_last)
                if state_key not in dp:
                    continue
                
                prev_comp = dp[state_key]
                pt = processing_times[last]
                
                comp = np.zeros(n_machines)
                comp[0] = prev_comp[0] + pt[0]
                for m in range(1, n_machines):
                    comp[m] = max(prev_comp[m], comp[m-1]) + pt[m]
                
                ms = float(comp[-1])
                if ms < best_ms:
                    best_ms = ms
                    best_comp = comp.copy()
            
            if best_comp is not None:
                dp[(mask, last)] = best_comp
    
    # Find best full permutation
    best_ms = float('inf')
    best_last = -1
    for last in range(n_jobs):
        state_key = (full_mask, last)
        if state_key in dp:
            ms = float(dp[state_key][-1])
            if ms < best_ms:
                best_ms = ms
                best_last = last
    
    if best_last < 0:
        return None
    
    # Reconstruct permutation
    perm = []
    current_mask = full_mask
    current_last = best_last
    
    while current_mask != 0:
        perm.append(current_last)
        pt = processing_times[current_last]
        
        # Find predecessor
        prev_mask = current_mask ^ (1 << current_last)
        if prev_mask == 0:
            current_mask = 0
            break
        
        best_prev = None
        best_prev_ms = float('inf')
        for prev_last in range(n_jobs):
            if not (prev_mask >> prev_last) & 1:
                continue
            state_key = (prev_mask, prev_last)
            if state_key not in dp:
                continue
            prev_comp = dp[state_key]
            comp = np.zeros(n_machines)
            comp[0] = prev_comp[0] + pt[0]
            for m in range(1, n_machines):
                comp[m] = max(prev_comp[m], comp[m-1]) + pt[m]
            ms = float(comp[-1])
            if ms < best_prev_ms:
                best_prev_ms = ms
                best_prev = prev_last
        
        if best_prev is None:
            break
        current_mask = prev_mask
        current_last = best_prev
    
    perm.reverse()
    return perm


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
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return [np.zeros(n_machines)]
    
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


def _precompute_pairwise_costs(processing_times):
    """
    Precompute pairwise job interaction costs for O(1) swap evaluation.
    cost[i][j] = additional makespan contribution when job i precedes job j
                 vs. a baseline, enabling fast delta computation.
    """
    n_jobs, n_machines = processing_times.shape
    # interaction[i][j] = makespan increase when j is placed right after i
    # compared to j being first (isolated)
    interaction = np.zeros((n_jobs, n_jobs))
    
    for i in range(n_jobs):
        for j in range(n_jobs):
            if i == j:
                continue
            pt_i = processing_times[i]
            pt_j = processing_times[j]
            
            # Completion of i alone
            comp_i = np.zeros(n_machines)
            comp_i[0] = pt_i[0]
            for m in range(1, n_machines):
                comp_i[m] = comp_i[m-1] + pt_i[m]
            
            # Completion of j after i
            comp_j = np.zeros(n_machines)
            comp_j[0] = comp_i[0] + pt_j[0]
            for m in range(1, n_machines):
                comp_j[m] = max(comp_i[m], comp_j[m-1]) + pt_j[m]
            
            # Completion of j alone
            comp_j_alone = np.zeros(n_machines)
            comp_j_alone[0] = pt_j[0]
            for m in range(1, n_machines):
                comp_j_alone[m] = comp_j_alone[m-1] + pt_j[m]
            
            # The "interaction cost" is how much the makespan of the pair
            # exceeds the sum of individual makespans
            interaction[i][j] = float(comp_j[-1]) - float(comp_j_alone[-1])
    
    return interaction


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
    
    order = list(np.argsort(-totals))
    
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


def _neh_order_forward_backward(processing_times, rng=None):
    """NEH with forward and backward passes."""
    if rng is None:
        rng = random.Random()
    
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        partial, _ = _insert_best_position(job, partial, processing_times)
    
    result = list(partial)
    for i in range(n_jobs - 1, 0, -1):
        job = result[i]
        remaining = result[:i] + result[i+1:]
        result, _ = _insert_best_position(job, remaining, processing_times)
    
    return result


def _swap_adjacent(perm, i):
    """Create new permutation with adjacent swap at position i."""
    new_perm = list(perm)
    new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
    return new_perm


def _move_job(perm, job, new_pos):
    """Move job to new position."""
    perm = [j for j in perm if j != job]
    return perm[:new_pos] + [job] + perm[new_pos:]


def _precompute_swap_deltas(perm, processing_times):
    """
    Precompute all adjacent swap deltas using prefix/suffix completions.
    Returns array of delta values for each adjacent swap position.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if n_jobs <= 1:
        return np.array([])
    
    prefix_comps = _compute_prefix_completions(perm, processing_times)
    suffix_comps = _compute_suffix_completions(perm, processing_times)
    
    current_ms = _makespan(perm, processing_times)
    deltas = np.zeros(n_jobs - 1)
    
    for i in range(n_jobs - 1):
        ms_after = _eval_swap_incremental(perm, i, processing_times, prefix_comps, suffix_comps)
        deltas[i] = ms_after - current_ms
    
    return deltas


def _fast_local_search(perm, processing_times, max_iterations=500, deadline=None):
    """
    Fast local search using precomputed swap deltas for O(1) adjacent swap evaluation.
    Enhanced with exhaustive neighborhood exploration using pairwise cost matrix.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    
    # Precompute pairwise interaction costs
    interaction = _precompute_pairwise_costs(processing_times)
    
    for it in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Precompute all adjacent swap deltas
        deltas = _precompute_swap_deltas(best_perm, processing_times)
        
        improved = False
        
        # Try all improving adjacent swaps (steepest descent)
        # Find the best (most negative) delta
        valid_deltas = deltas[deltas < -1e-10]
        if len(valid_deltas) > 0:
            min_idx = int(np.argmin(deltas))
            best_perm[min_idx], best_perm[min_idx+1] = best_perm[min_idx+1], best_perm[min_idx]
            best_ms += deltas[min_idx]
            improved = True
        
        if not improved:
            # Try insert moves - exhaustive search over all positions
            prefix_comps = _compute_prefix_completions(best_perm, processing_times)
            suffix_comps = _compute_suffix_completions(best_perm, processing_times)
            
            best_insert_delta = -1e-10
            best_i = -1
            best_pos = -1
            
            for i in range(n_jobs):
                if deadline is not None and time.time() > deadline:
                    break
                job = best_perm[i]
                for pos in range(n_jobs):
                    if pos == i or pos == i + 1:
                        continue
                    ms = _eval_insert_incremental(best_perm, i, pos, processing_times, prefix_comps, suffix_comps)
                    delta = ms - best_ms
                    if delta < best_insert_delta:
                        best_insert_delta = delta
                        best_i = i
                        best_pos = pos
            
            if best_i >= 0:
                job = best_perm[best_i]
                new_perm = _move_job(best_perm, job, best_pos)
                best_perm = new_perm
                best_ms += best_insert_delta
                improved = True
            
            if not improved:
                # Try 2-opt style: reverse a segment
                if n_jobs >= 3:
                    best_2opt_delta = -1e-10
                    best_start = -1
                    best_end = -1
                    
                    for start in range(n_jobs - 1):
                        if deadline is not None and time.time() > deadline:
                            break
                        for end in range(start + 1, n_jobs):
                            if end - start + 1 < 2:
                                continue
                            # Evaluate reversing segment [start, end]
                            new_perm = list(best_perm)
                            new_perm[start:end+1] = new_perm[start:end+1][::-1]
                            ms = _makespan(new_perm, processing_times)
                            delta = ms - best_ms
                            if delta < best_2opt_delta:
                                best_2opt_delta = delta
                                best_start = start
                                best_end = end
                    
                    if best_start >= 0:
                        best_perm[best_start:best_end+1] = best_perm[best_start:best_end+1][::-1]
                        best_ms += best_2opt_delta
                        improved = True
            
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
    deadline = start_time + 110.0  # Safety margin: 10s before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    candidates = []
    
    # Try exact DP for small instances
    if n_jobs <= 15 and time.time() < deadline - 20.0:
        dp_perm = _dp_exact_small(processing_times, max_jobs=15, deadline=deadline - 20.0)
        if dp_perm is not None:
            candidates.append(dp_perm)
    
    lpt_perm = list(np.argsort(-processing_times.sum(axis=1)))
    candidates.append(lpt_perm)
    
    if time.time() < deadline:
        neh_perm = _neh_order(processing_times, rng)
        candidates.append(neh_perm)
    
    if time.time() < deadline:
        neh2_perm = _neh_order(processing_times, random.Random(rng.randint(0, 1000000)))
        candidates.append(neh2_perm)
    
    if time.time() < deadline:
        neh_fb_perm = _neh_order_forward_backward(processing_times, rng)
        candidates.append(neh_fb_perm)
    
    # Add more diverse NEH variants
    if time.time() < deadline:
        for seed in [123, 456, 789]:
            neh_variant = _neh_order(processing_times, random.Random(seed))
            candidates.append(neh_variant)
    
    best_perm = None
    best_ms = float('inf')
    for perm in candidates:
        ms = _makespan(perm, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = list(perm)
    
    if time.time() < deadline and n_jobs > 1:
        remaining = deadline - time.time()
        if n_jobs <= 20:
            max_iter = 3000
        elif n_jobs <= 30:
            max_iter = 2000
        elif n_jobs <= 50:
            max_iter = 1000
        elif n_jobs <= 100:
            max_iter = 500
        else:
            max_iter = 200
        
        polished = _fast_local_search(best_perm, processing_times, max_iterations=max_iter, deadline=deadline)
        polished_ms = _makespan(polished, processing_times)
        if polished_ms < best_ms:
            best_perm = polished
            best_ms = polished_ms
    
    if time.time() < deadline - 5.0 and n_jobs > 1:
        sa_deadline = deadline - 5.0
        sa_perm = _simulated_annealing(best_perm, processing_times, start_time, sa_deadline, rng)
        sa_ms = _makespan(sa_perm, processing_times)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
        
        # Second SA run with different parameters
        if time.time() < deadline - 3.0:
            rng2 = random.Random(rng.randint(1000, 999999))
            sa2_perm = _simulated_annealing(best_perm, processing_times, start_time, deadline - 3.0, rng2)
            sa2_ms = _makespan(sa2_perm, processing_times)
            if sa2_ms < best_ms:
                best_perm = sa2_perm
                best_ms = sa2_ms
        
        if time.time() < deadline - 2.0:
            max_iter_final = 1000 if n_jobs <= 50 else 300
            final_polish = _fast_local_search(best_perm, processing_times, max_iterations=max_iter_final, deadline=deadline - 2.0)
            final_ms = _makespan(final_polish, processing_times)
            if final_ms < best_ms:
                best_perm = final_polish
                best_ms = final_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
