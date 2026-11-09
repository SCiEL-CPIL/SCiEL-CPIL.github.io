"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    # Use vectorized computation where possible
    # C[j][m] = completion time of job j on machine m
    # We can compute this incrementally
    completion = np.zeros((n_jobs, n_machines), dtype=np.float64)
    
    for j in range(n_jobs):
        job_idx = order[j]
        for m in range(n_machines):
            p = processing_times[job_idx, m]
            if m == 0:
                if j == 0:
                    completion[j, m] = p
                else:
                    completion[j, m] = completion[j-1, m] + p
            else:
                if j == 0:
                    completion[j, m] = completion[j, m-1] + p
                else:
                    completion[j, m] = max(completion[j-1, m], completion[j, m-1]) + p
    
    return completion[-1, -1]


def _makespan_fast(processing_times, order):
    """Compute makespan using numpy for speed."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    # Compute completion times row by row
    # C[j][m] = max(C[j-1][m], C[j][m-1]) + P[j][m]
    # For machine 0: C[j][0] = C[j-1][0] + P[j][0]
    # For job 0: C[0][m] = C[0][m-1] + P[0][m]
    
    # Use the fact that we only need the last row and last column
    # Actually, let's use a vectorized approach with cumulative sums where possible
    
    # For each machine, the completion time depends on previous machine and previous job
    # This is inherently sequential, but we can optimize the inner loop
    
    # Pre-extract processing times in order
    pt = processing_times[order]  # (n_jobs, n_machines)
    
    # C[j][m] = max(C[j-1][m], C[j][m-1]) + pt[j][m]
    # We can compute this iteratively
    if n_machines == 1:
        return float(np.sum(pt[:, 0]))
    
    # Start with first machine: cumulative sum
    c_prev = np.cumsum(pt[:, 0])  # Completion times for machine 0
    
    for m in range(1, n_machines):
        # For machine m, C[j][m] = max(C[j-1][m], C[j][m-1]) + pt[j][m]
        # C[j][m-1] is c_prev[j]
        # C[j-1][m] is the previous value in the current machine's completion array
        # This is a sequential dependency within the machine
        
        c_curr = np.empty(n_jobs, dtype=np.float64)
        c_curr[0] = c_prev[0] + pt[0, m]
        for j in range(1, n_jobs):
            c_curr[j] = max(c_curr[j-1], c_prev[j]) + pt[j, m]
        c_prev = c_curr
    
    return float(c_prev[-1])


def _neh_rule(processing_times):
    """NEH heuristic for flow-shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    
    # Sort jobs by total processing time in descending order
    total_pt = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_pt)
    
    # Build the permutation incrementally
    permutation = [sorted_jobs[0]]
    
    for job in sorted_jobs[1:]:
        # Try inserting the job at each position
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(permutation) + 1):
            new_perm = permutation[:pos] + [job] + permutation[pos:]
            ms = _makespan(processing_times, new_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        permutation = permutation[:best_pos] + [job] + permutation[best_pos:]
    
    return permutation


def _local_search_swap(processing_times, order, deadline=None):
    """Local search with adjacent swaps only (fast)."""
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order), _makespan_fast(processing_times, order)
    
    current_order = list(order)
    current_ms = _makespan_fast(processing_times, current_order)
    
    improved = True
    while improved:
        improved = False
        if deadline is not None and time.time() > deadline:
            break
        for i in range(n_jobs - 1):
            if deadline is not None and time.time() > deadline:
                break
            new_order = current_order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            new_ms = _makespan_fast(processing_times, new_order)
            if new_ms < current_ms:
                current_order = new_order
                current_ms = new_ms
                improved = True
    
    return current_order, current_ms


def _local_search_insert(processing_times, order, deadline=None):
    """Local search with insert moves (best improvement per job)."""
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order), _makespan_fast(processing_times, order)
    
    current_order = list(order)
    current_ms = _makespan_fast(processing_times, current_order)
    
    improved = True
    while improved:
        improved = False
        if deadline is not None and time.time() > deadline:
            break
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
            job = current_order[i]
            # Remove job
            rest = current_order[:i] + current_order[i+1:]
            best_pos = -1
            best_ms = current_ms
            
            # Try inserting at each position
            for pos in range(len(rest) + 1):
                if pos == i or pos == i - 1:
                    continue  # Skip original position
                new_order = rest[:pos] + [job] + rest[pos:]
                new_ms = _makespan_fast(processing_times, new_order)
                if new_ms < best_ms:
                    best_ms = new_ms
                    best_pos = pos
            
            if best_pos != -1:
                rest = current_order[:i] + current_order[i+1:]
                current_order = rest[:best_pos] + [job] + rest[best_pos:]
                current_ms = best_ms
                improved = True
    
    return current_order, current_ms


def _local_search_2opt(processing_times, order, deadline=None):
    """Local search with 2-opt (swap any two jobs, best improvement)."""
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order), _makespan_fast(processing_times, order)
    
    current_order = list(order)
    current_ms = _makespan_fast(processing_times, current_order)
    
    improved = True
    while improved:
        improved = False
        if deadline is not None and time.time() > deadline:
            break
        best_i, best_j = -1, -1
        best_ms = current_ms
        
        for i in range(n_jobs - 1):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(i + 1, n_jobs):
                if deadline is not None and time.time() > deadline:
                    break
                new_order = current_order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_ms = _makespan_fast(processing_times, new_order)
                if new_ms < best_ms:
                    best_ms = new_ms
                    best_i, best_j = i, j
        
        if best_i != -1:
            current_order[best_i], current_order[best_j] = current_order[best_j], current_order[best_i]
            current_ms = best_ms
            improved = True
    
    return current_order, current_ms


def _simulated_annealing(processing_times, initial_order, start_temp=50.0, end_temp=0.01, 
                          cooling_rate=0.97, max_iter=2000, deadline=None):
    """Simulated annealing with adjacent swaps, insert moves, and 2-opt swaps."""
    n_jobs = len(initial_order)
    if n_jobs < 2:
        return list(initial_order), _makespan_fast(processing_times, initial_order)
    
    current_order = list(initial_order)
    current_ms = _makespan_fast(processing_times, current_order)
    best_order = list(current_order)
    best_ms = current_ms
    
    temp = start_temp
    iter_count = 0
    
    while temp > end_temp and iter_count < max_iter:
        iter_count += 1
        
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate a random move: adjacent swap (40%), insert (40%), 2-opt (20%)
        move_type = np.random.random()
        
        if move_type < 0.4:
            # Adjacent swap
            i = np.random.randint(0, n_jobs - 1)
            j = i + 1
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
        elif move_type < 0.8:
            # Insert move
            i = np.random.randint(0, n_jobs)
            job = current_order[i]
            rest = current_order[:i] + current_order[i+1:]
            pos = np.random.randint(0, len(rest) + 1)
            if pos == i or pos == i - 1:
                pos = (pos + 1) % (len(rest) + 1)
            new_order = rest[:pos] + [job] + rest[pos:]
        else:
            # 2-opt swap
            i = np.random.randint(0, n_jobs)
            j = np.random.randint(0, n_jobs)
            if i == j:
                j = (j + 1) % n_jobs
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
        
        new_ms = _makespan_fast(processing_times, new_order)
        delta = new_ms - current_ms
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_order = new_order
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = list(current_order)
        
        temp *= cooling_rate
    
    return best_order, best_ms


def _multi_start_local_search(processing_times, initial_order, deadline, n_starts=5):
    """Multi-start local search with diverse initial solutions."""
    n_jobs = len(initial_order)
    if n_jobs < 2:
        return list(initial_order), _makespan_fast(processing_times, initial_order)
    
    best_order = list(initial_order)
    best_ms = _makespan_fast(processing_times, best_order)
    
    # Generate diverse starting points
    starts = [list(initial_order)]
    
    # Add NEH-based variants
    total_pt = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_pt)
    starts.append(list(sorted_jobs))
    
    # Add reverse NEH
    starts.append(list(sorted_jobs[::-1]))
    
    # Add random starts
    for _ in range(n_starts - 3):
        if time.time() > deadline:
            break
        rand_perm = list(range(n_jobs))
        np.random.shuffle(rand_perm)
        starts.append(rand_perm)
    
    # For each start, run local search
    for start in starts:
        if time.time() > deadline:
            break
        
        # Run swap local search
        order, ms = _local_search_swap(processing_times, start, deadline)
        if ms < best_ms:
            best_ms = ms
            best_order = order
        
        # Run insert local search on the result
        if time.time() < deadline - 5:
            order, ms = _local_search_insert(processing_times, best_order, deadline)
            if ms < best_ms:
                best_ms = ms
                best_order = order
    
    return best_order, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    
    # Handle edge cases
    if n_jobs <= 1:
        return list(range(n_jobs))
    if n_machines == 0:
        return list(range(n_jobs))
    
    # Start with NEH heuristic
    neh_order = _neh_rule(processing_times)
    best_order = list(neh_order)
    best_ms = _makespan_fast(processing_times, best_order)
    
    # Quick local search on NEH: swap then insert
    if time.time() < deadline - 40:
        ls_order, ls_ms = _local_search_swap(processing_times, best_order, deadline)
        if ls_ms < best_ms:
            best_order = ls_order
            best_ms = ls_ms
        if time.time() < deadline - 30:
            ls_order, ls_ms = _local_search_insert(processing_times, best_order, deadline)
            if ls_ms < best_ms:
                best_order = ls_order
                best_ms = ls_ms
    
    # Multi-start local search with diverse neighborhoods
    if time.time() < deadline - 30:
        ms_deadline = deadline - 20
        ms_order, ms_ms = _multi_start_local_search(
            processing_times, best_order, ms_deadline, n_starts=5
        )
        if ms_ms < best_ms:
            best_order = ms_order
            best_ms = ms_ms
    
    # Simulated annealing for further improvement (longer run)
    if time.time() < deadline - 10:
        sa_deadline = deadline - 5
        sa_order, sa_ms = _simulated_annealing(processing_times, best_order, 
                                                start_temp=50.0, end_temp=0.01,
                                                cooling_rate=0.97, max_iter=2000,
                                                deadline=sa_deadline)
        if sa_ms < best_ms:
            best_order = sa_order
            best_ms = sa_ms
    
    # Final high-precision polish: alternate between all move types
    if time.time() < deadline - 2:
        polish_deadline = deadline - 0.5
        for _ in range(5):  # Five polish passes
            if time.time() > polish_deadline:
                break
            # Swap polish
            order, ms = _local_search_swap(processing_times, best_order, polish_deadline)
            if ms < best_ms:
                best_order = order
                best_ms = ms
            # Insert polish
            if time.time() < polish_deadline:
                order, ms = _local_search_insert(processing_times, best_order, polish_deadline)
                if ms < best_ms:
                    best_order = order
                    best_ms = ms
            # 2-opt polish
            if time.time() < polish_deadline:
                order, ms = _local_search_2opt(processing_times, best_order, polish_deadline)
                if ms < best_ms:
                    best_order = order
                    best_ms = ms
    
    return best_order
# EVOLVE-BLOCK-END
