"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0 or n_machines == 0:
        return 0.0
    
    completion = np.zeros(n_machines)
    for job_idx in perm:
        for m in range(n_machines):
            if m == 0:
                completion[m] += processing_times[job_idx, m]
            else:
                completion[m] = max(completion[m], completion[m - 1]) + processing_times[job_idx, m]
    return completion[-1]


def _makespan_fast(perm, processing_times):
    """Fast makespan computation for a given permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0 or n_machines == 0:
        return 0.0
    
    pt = processing_times
    completion = [0.0] * n_machines
    
    for job_idx in perm:
        row = pt[job_idx]
        for m in range(n_machines):
            if m == 0:
                completion[m] += row[m]
            else:
                prev = completion[m - 1]
                cur = completion[m]
                if prev > cur:
                    completion[m] = prev + row[m]
                else:
                    completion[m] = cur + row[m]
    return completion[-1]


def _compute_completion_times(perm, processing_times):
    """Compute completion times for all machines at each position."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0 or n_machines == 0:
        return [], []
    
    # comp[job_pos][m] = completion time of job at position job_pos on machine m
    comp = []
    current_comp = [0.0] * n_machines
    for job_idx in perm:
        row = processing_times[job_idx]
        new_comp = [0.0] * n_machines
        for m in range(n_machines):
            if m == 0:
                new_comp[m] = current_comp[m] + row[m]
            else:
                prev = new_comp[m - 1]
                cur = current_comp[m]
                if prev > cur:
                    new_comp[m] = prev + row[m]
                else:
                    new_comp[m] = cur + row[m]
        comp.append(new_comp)
        current_comp = new_comp
    return comp, current_comp


def _eval_swap_incremental(comp, perm, processing_times, i, n_machines):
    """Evaluate makespan after swapping positions i and i+1 using incremental computation.
    Returns the new makespan."""
    n_jobs = len(perm)
    if i >= n_jobs - 1:
        return float('inf')
    
    job_a = perm[i]
    job_b = perm[i + 1]
    
    # Get completion times before position i
    if i == 0:
        prev_comp = [0.0] * n_machines
    else:
        prev_comp = comp[i - 1]
    
    # Process job_b (which moves to position i) then job_a (which moves to position i+1)
    row_b = processing_times[job_b]
    row_a = processing_times[job_a]
    
    comp_b = [0.0] * n_machines
    for m in range(n_machines):
        if m == 0:
            comp_b[m] = prev_comp[m] + row_b[m]
        else:
            prev_val = comp_b[m - 1]
            cur_val = prev_comp[m]
            if prev_val > cur_val:
                comp_b[m] = prev_val + row_b[m]
            else:
                comp_b[m] = cur_val + row_b[m]
    
    comp_a = [0.0] * n_machines
    for m in range(n_machines):
        if m == 0:
            comp_a[m] = comp_b[m] + row_a[m]
        else:
            prev_val = comp_a[m - 1]
            cur_val = comp_b[m]
            if prev_val > cur_val:
                comp_a[m] = prev_val + row_a[m]
            else:
                comp_a[m] = cur_val + row_a[m]
    
    # Now we need to propagate from position i+2 onwards
    # The completion time at position i+1 after swap is comp_a
    # We need to recompute from position i+2 to the end
    
    if i + 2 >= n_jobs:
        return comp_a[-1]
    
    # Recompute suffix
    cur_comp = comp_a
    for pos in range(i + 2, n_jobs):
        job_idx = perm[pos]
        row = processing_times[job_idx]
        new_comp = [0.0] * n_machines
        for m in range(n_machines):
            if m == 0:
                new_comp[m] = cur_comp[m] + row[m]
            else:
                prev_val = new_comp[m - 1]
                cur_val = cur_comp[m]
                if prev_val > cur_val:
                    new_comp[m] = prev_val + row[m]
                else:
                    new_comp[m] = cur_val + row[m]
        cur_comp = new_comp
    
    return cur_comp[-1]


def _neh_order(processing_times):
    """NEH heuristic: insert jobs one by one in order of decreasing total processing time."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(new_perm, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _adjacent_swap_neighborhood(perm, processing_times, n_iterations=2000, deadline=None):
    """Local search using adjacent swaps with incremental evaluation."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    current_perm = list(perm)
    n_machines = processing_times.shape[1]
    
    # Precompute completion times
    comp, final_comp = _compute_completion_times(current_perm, processing_times)
    current_ms = final_comp[-1]
    
    improved = True
    while improved and n_iterations > 0:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n_jobs - 1):
            # Evaluate swap using incremental method
            new_ms = _eval_swap_incremental(comp, current_perm, processing_times, i, n_machines)
            if new_ms < current_ms - 1e-12:
                # Perform the swap
                current_perm[i], current_perm[i + 1] = current_perm[i + 1], current_perm[i]
                current_ms = new_ms
                improved = True
                n_iterations -= 1
                if n_iterations <= 0:
                    break
                # Recompute completion times after the swap
                comp, final_comp = _compute_completion_times(current_perm, processing_times)
                # Restart scan from the beginning
                break
    
    return current_perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using NEH + local search.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 145s out of 150s limit
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: NEH constructive heuristic
    perm = _neh_order(processing_times)
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, processing_times)
    
    # Step 2: Local search with adjacent swaps (incremental)
    perm = _adjacent_swap_neighborhood(perm, processing_times, n_iterations=50000, deadline=deadline)
    ms = _makespan_fast(perm, processing_times)
    if ms < best_ms:
        best_perm = list(perm)
        best_ms = ms
    
    # Step 3: If time remains, do more local search passes with perturbations
    elapsed = time.time() - start_time
    if elapsed < deadline:
        rng = np.random.default_rng(42)
        for restart in range(100):
            if time.time() > deadline:
                break
            
            # Perturb the best permutation
            perturbed = list(best_perm)
            i = rng.integers(0, n_jobs - 1)
            j = rng.integers(i + 1, n_jobs)
            perturbed[i:j + 1] = perturbed[i:j + 1][::-1]
            
            # Local search from perturbed start
            improved_perm = _adjacent_swap_neighborhood(perturbed, processing_times, n_iterations=5000, deadline=deadline)
            improved_ms = _makespan_fast(improved_perm, processing_times)
            if improved_ms < best_ms:
                best_perm = list(improved_perm)
                best_ms = improved_ms
    
    return best_perm
# EVOLVE-BLOCK-END
