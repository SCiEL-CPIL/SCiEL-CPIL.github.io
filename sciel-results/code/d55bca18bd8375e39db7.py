"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    c = np.zeros(n_machines)
    for job_idx in order:
        p_row = processing_times[job_idx]
        prev_c = 0.0
        for j in range(n_machines):
            c_val = c[j]
            if prev_c > c_val:
                c_val = prev_c
            c_val += p_row[j]
            c[j] = c_val
            prev_c = c_val
    return c[-1]

def _completion_matrix(processing_times, order):
    """Compute full completion time matrix C[job_pos][machine]."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    C = np.zeros((n_jobs, n_machines))
    for i, job_idx in enumerate(order):
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            if i == 0:
                C[i, j] = C[i, j-1] + p_row[j] if j > 0 else p_row[j]
            elif j == 0:
                C[i, j] = C[i-1, 0] + p_row[j]
            else:
                C[i, j] = max(C[i-1, j], C[i, j-1]) + p_row[j]
    return C

def _makespan_from_C(C):
    """Get makespan from completion matrix."""
    if len(C) == 0:
        return 0.0
    return C[-1, -1]

def _critical_path_length(processing_times, order):
    """
    Estimate critical path length for a given order using DP on the grid.
    This is equivalent to makespan but computed via path formulation.
    """
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    # dp[j] = earliest completion time at machine j for the last job
    dp = np.zeros(n_machines)
    for job_idx in order:
        p_row = processing_times[job_idx]
        new_dp = np.zeros(n_machines)
        prev = 0.0
        for j in range(n_machines):
            # Job can start on machine j after:
            # 1. Previous job finishes on machine j (dp[j])
            # 2. Same job finishes on machine j-1 (prev)
            start = max(dp[j], prev)
            new_dp[j] = start + p_row[j]
            prev = new_dp[j]
        dp = new_dp
    return dp[-1]

def _makespan_from_c(processing_times, order, c_start):
    """Compute makespan starting from a given completion time vector."""
    if len(order) == 0:
        return c_start[-1] if len(c_start) > 0 else 0.0
    n_machines = processing_times.shape[1]
    c = c_start.copy()
    for job_idx in order:
        p_row = processing_times[job_idx]
        prev_c = 0.0
        for j in range(n_machines):
            c_val = c[j]
            if prev_c > c_val:
                c_val = prev_c
            c_val += p_row[j]
            c[j] = c_val
            prev_c = c_val
    return c[-1]

def _neh_construct(processing_times, sort_key):
    """
    NEH constructive heuristic with a given sort key function.
    Returns the best order found by inserting jobs in sorted order at best position.
    """
    n_jobs = processing_times.shape[0]
    totals = processing_times.sum(axis=1)
    
    # Generate sort order based on key
    if sort_key == 'total_desc':
        sorted_jobs = list(np.argsort(-totals))
    elif sort_key == 'total_asc':
        sorted_jobs = list(np.argsort(totals))
    elif sort_key == 'first_machine_desc':
        sorted_jobs = list(np.argsort(-processing_times[:, 0]))
    elif sort_key == 'last_machine_desc':
        sorted_jobs = list(np.argsort(-processing_times[:, -1]))
    elif sort_key == 'mean_desc':
        sorted_jobs = list(np.argsort(-processing_times.mean(axis=1)))
    elif sort_key == 'variance_asc':
        sorted_jobs = list(np.argsort(processing_times.var(axis=1)))
    elif sort_key == 'range_desc':
        ranges = processing_times.max(axis=1) - processing_times.min(axis=1)
        sorted_jobs = list(np.argsort(-ranges))
    else:
        sorted_jobs = list(np.argsort(-totals))
    
    current_order = [sorted_jobs[0]]
    for job in sorted_jobs[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current_order) + 1):
            new_order = current_order[:pos] + [job] + current_order[pos:]
            ms = _makespan(processing_times, new_order)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current_order.insert(best_pos, job)
    
    return current_order

def _first_improvement_ls(processing_times, order, ms, deadline):
    """
    First-improvement local search using insertion moves.
    Returns improved order and makespan.
    """
    n = len(order)
    if n < 2:
        return order, ms
    
    current_order = list(order)
    current_ms = ms
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            job = current_order[i]
            temp_order = current_order[:i] + current_order[i+1:]
            
            for pos in range(len(temp_order)):
                if pos == i:
                    continue
                new_order = temp_order[:pos] + [job] + temp_order[pos:]
                new_ms = _makespan(processing_times, new_order)
                if new_ms < current_ms:
                    current_order = new_order
                    current_ms = new_ms
                    improved = True
                    break
    
    return current_order, current_ms

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    deadline = start_time + 148  # 2s safety margin
    
    # --- Phase 1: Multi-variant NEH Constructive Heuristic ---
    # Try multiple sorting keys to generate diverse initial solutions
    sort_keys = ['total_desc', 'first_machine_desc', 'last_machine_desc', 'mean_desc', 'variance_asc']
    
    best_order = None
    best_ms = float('inf')
    
    for key in sort_keys:
        if time.time() > deadline:
            break
        candidate_order = _neh_construct(processing_times, key)
        candidate_ms = _makespan(processing_times, candidate_order)
        if candidate_ms < best_ms:
            best_ms = candidate_ms
            best_order = list(candidate_order)
    
    # --- Phase 2: First-Improvement Local Search (fast, frees up budget) ---
    if time.time() < deadline:
        best_order, best_ms = _first_improvement_ls(processing_times, best_order, best_ms, deadline)
    
    # --- Phase 3: Simulated Annealing ---
    if time.time() < deadline:
        current = list(best_order)
        current_ms = best_ms
        T = max(1.0, best_ms * 0.02)  # Slightly higher initial temperature
        cooling = 0.9995  # Slower cooling for more exploration
        min_T = 1e-4
        
        while T > min_T and time.time() < deadline:
            n = len(current)
            if n < 2:
                break
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                T *= cooling
                continue
            
            # Use insertion move for larger neighborhood
            job = current[i]
            temp = current[:i] + current[i+1:]
            if i < j:
                j -= 1
            new_order = temp[:j] + [job] + temp[j:]
            
            new_ms = _makespan(processing_times, new_order)
            delta = new_ms - current_ms
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current = new_order
                current_ms = new_ms
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_order = list(current)
            
            T *= cooling
    
    # --- Phase 4: Final Polish (First-Improvement Local Search) ---
    if time.time() < deadline:
        # Multiple rounds of first-improvement LS
        for _ in range(5):
            if time.time() > deadline:
                break
            prev_ms = best_ms
            best_order, best_ms = _first_improvement_ls(processing_times, best_order, best_ms, deadline)
            if best_ms >= prev_ms:
                break
    
    return best_order
# EVOLVE-BLOCK-END
