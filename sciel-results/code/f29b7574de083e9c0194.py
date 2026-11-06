"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan_from_order(pt, order):
    """Compute makespan for a given job order. O(n*m) using cumulative sums."""
    n_jobs = len(order)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    # Build the sequence of jobs
    seq = pt[order]  # shape (n_jobs, n_machines)
    # Flow shop makespan: C[i][j] = max(C[i-1][j], C[i][j-1]) + p[i][j]
    # Vectorized: use cumulative approach
    # C[j] for machine j after processing all jobs up to current
    # We can compute iteratively over machines
    # Actually, the standard way:
    # completion[i][j] = max(completion[i-1][j], completion[i][j-1]) + p[i][j]
    # For makespan, we only need the last machine's completion for the last job
    # But we need the full table. Let's do it efficiently.
    
    # Use a single array for the current row of completions
    # comp[j] = completion time on machine j for the current job
    # After processing job i: comp_new[j] = max(comp_old[j], comp_new[j-1]) + p[i][j]
    
    comp = np.zeros(n_machines)
    for i in range(n_jobs):
        job_pt = seq[i]
        prev = 0.0
        for j in range(n_machines):
            prev = max(comp[j], prev) + job_pt[j]
            comp[j] = prev
    return comp[-1]


def _neh_initialization(pt):
    """NEH heuristic for flow shop. O(n^2 * m)."""
    n_jobs, n_machines = pt.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Sort jobs by total processing time descending
    totals = pt.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    # Insert jobs one by one into the best position
    order = [sorted_jobs[0]]
    for idx in range(1, n_jobs):
        job = sorted_jobs[idx]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            # Try inserting job at position pos
            new_order = order[:pos] + [job] + order[pos:]
            ms = _makespan_from_order(pt, new_order)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order = order[:best_pos] + [job] + order[best_pos:]
    
    return order


def _vns_search(pt, initial_order, time_limit=145.0):
    """
    Variable Neighborhood Search with insertion neighborhoods of increasing radius.
    """
    n_jobs = len(initial_order)
    n_machines = pt.shape[1]
    
    if n_jobs <= 1:
        return initial_order
    
    best_order = list(initial_order)
    best_ms = _makespan_from_order(pt, best_order)
    
    start_time = time.time()
    max_k = min(n_jobs - 1, 50)  # Maximum neighborhood radius
    
    # VNS: iterate over neighborhood structures k = 1, 2, ..., max_k
    # For each k, perform random insertions with |i-j| <= k until local optimum
    # Then move to next k if no improvement, or reset k if improvement found
    
    k = 1
    while k <= max_k:
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Shake: generate a random perturbation
        # For VNS, we typically do a fixed shake then local search
        # Here we do: for current k, try random insertions within radius k
        # If improvement found, reset k=1. If no improvement after some tries, increase k.
        
        improved = False
        attempts = 0
        max_attempts_per_k = max(10, n_jobs * 2)
        
        while attempts < max_attempts_per_k:
            if time.time() - start_time > time_limit:
                return best_order
            attempts += 1
            
            # Random insertion: pick a random job and move it to a random position within radius k
            i = np.random.randint(0, n_jobs)
            # Choose destination j such that |i - j| <= k
            max_shift = min(k, n_jobs - 1)
            j_offset = np.random.randint(-max_shift, max_shift + 1)
            j = i + j_offset
            if j < 0 or j >= n_jobs:
                continue
            if i == j:
                continue
            
            # Perform insertion: remove job at i, insert at j
            job = best_order[i]
            new_order = best_order[:i] + best_order[i+1:]
            # Adjust j if needed
            if i < j:
                j -= 1
            new_order = new_order[:j] + [job] + new_order[j:]
            
            # Evaluate
            ms = _makespan_from_order(pt, new_order)
            if ms < best_ms - 1e-12:
                best_order = new_order
                best_ms = ms
                improved = True
                # Reset k to 1 since we found improvement
                k = 1
                break
        
        if not improved:
            k += 1
    
    return best_order


def _local_search_polish(pt, order, time_limit=145.0):
    """
    Final polish with first-improvement adjacent swaps and insertions.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return order
    
    best_order = list(order)
    best_ms = _makespan_from_order(pt, best_order)
    
    start_time = time.time()
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # Try all adjacent swaps
        for i in range(n_jobs - 1):
            if time.time() - start_time > time_limit:
                return best_order
            new_order = best_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            ms = _makespan_from_order(pt, new_order)
            if ms < best_ms - 1e-12:
                best_order = new_order
                best_ms = ms
                improved = True
                break
        
        # Try insertions: move each job to each other position
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                return best_order
            job = best_order[i]
            for j in range(n_jobs):
                if j == i or j == i + 1:
                    continue
                new_order = best_order[:i] + best_order[i+1:]
                if i < j:
                    j -= 1
                new_order = new_order[:j] + [job] + new_order[j:]
                ms = _makespan_from_order(pt, new_order)
                if ms < best_ms - 1e-12:
                    best_order = new_order
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
    
    return best_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow shop instance.
    Uses NEH initialization followed by Variable Neighborhood Search and local polish.
    """
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs = pt.shape[0]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Set a deadline guard: 150s wall-clock limit, use 145s as safety margin
    start_time = time.time()
    time_limit = 145.0
    
    # Step 1: NEH initialization
    order = _neh_initialization(pt)
    
    # Check time
    if time.time() - start_time > time_limit:
        return order
    
    # Step 2: VNS search
    order = _vns_search(pt, order, time_limit=time_limit)
    
    # Check time
    if time.time() - start_time > time_limit:
        return order
    
    # Step 3: Local search polish
    order = _local_search_polish(pt, order, time_limit=time_limit)
    
    return order
# EVOLVE-BLOCK-END
