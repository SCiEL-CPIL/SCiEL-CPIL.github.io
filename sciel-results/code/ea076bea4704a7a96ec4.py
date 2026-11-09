"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order.
    
    processing_times: shape (n_jobs, n_machines)
    order: list/array of job indices in sequence
    Returns: makespan (float)
    """
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    # completion[m] = completion time of last job on machine m
    completion = np.zeros(n_machines)
    
    for job in order:
        for m in range(n_machines):
            if m == 0:
                completion[m] += processing_times[job, m]
            else:
                # Machine m can start after machine m-1 finishes this job AND machine m is free
                completion[m] = max(completion[m], completion[m-1]) + processing_times[job, m]
    
    return completion[-1]


def _makespan_vectorized(processing_times, order):
    """Vectorized makespan computation for speed."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    # Extract submatrix in job order
    sub = processing_times[list(order), :]
    
    # DP: C[j, m] = completion time of job j on machine m
    # C[j, 0] = C[j-1, 0] + p[j, 0]
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
    C = np.zeros((n_jobs, n_machines))
    C[0, 0] = sub[0, 0]
    for j in range(1, n_jobs):
        C[j, 0] = C[j-1, 0] + sub[j, 0]
    for m in range(1, n_machines):
        C[0, m] = C[0, m-1] + sub[0, m]
        for j in range(1, n_jobs):
            C[j, m] = max(C[j-1, m], C[j, m-1]) + sub[j, m]
    
    return C[-1, -1]


def _neighborhood_insert(processing_times, order):
    """Generate all possible insert moves (remove job, insert at new position)."""
    n = len(order)
    neighbors = []
    for i in range(n):
        job = order[i]
        for j in range(n):
            if j == i:
                continue
            # Remove job from position i
            new_order = order[:i] + order[i+1:]
            # Insert at position j (adjust if j > i)
            insert_pos = j if j < i else j
            new_order = new_order[:insert_pos] + [job] + new_order[insert_pos:]
            neighbors.append(new_order)
    return neighbors


def _swap_neighbors(processing_times, order):
    """Generate all swap moves."""
    n = len(order)
    neighbors = []
    for i in range(n):
        for j in range(i+1, n):
            new_order = order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
            neighbors.append(new_order)
    return neighbors


def order_jobs(processing_times):
    """
    Tabu search for flow shop scheduling.
    
    Uses NEH heuristic for initial solution, then tabu search with
    insertion neighborhood to escape local optima.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    processing_times = np.asarray(processing_times, dtype=np.float64)
    
    # --- Step 1: NEH heuristic for initial solution ---
    # Sort by total processing time descending, then insert each job in best position
    totals = processing_times.sum(axis=1)
    ne_order = list(np.argsort(-totals))
    
    # Build solution by inserting jobs one by one in best position
    current = []
    for job in ne_order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            trial = current[:pos] + [job] + current[pos:]
            ms = _makespan_vectorized(processing_times, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
    
    best_order = current.copy()
    best_ms = _makespan_vectorized(processing_times, best_order)
    
    # --- Step 2: Tabu Search ---
    # Tabu list: store (job_removed, position_removed, position_inserted) for reverse moves
    tabu_list = []
    tabu_tenure = max(1, n_jobs // 2)
    max_iterations = 20000
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() > deadline:
            break
        
        # Generate all insertion moves
        best_neighbor = None
        best_neighbor_ms = float('inf')
        best_neighbor_key = None
        
        n = len(best_order)
        for i in range(n):
            job = best_order[i]
            for j in range(n):
                if j == i:
                    continue
                # Create new order by moving job from i to j
                new_order = best_order[:i] + best_order[i+1:]
                insert_pos = j if j < i else j
                new_order = new_order[:insert_pos] + [job] + new_order[insert_pos:]
                
                ms = _makespan_vectorized(processing_times, new_order)
                
                # Check if this move is tabu (reverse of a recent move)
                # Key: (job, from_pos, to_pos) - tabu if reverse was recent
                move_key = (job, i, j)
                is_tabu = False
                for tabu_entry in tabu_list:
                    # Reverse move: (job, j, i) was recently made
                    if tabu_entry[0] == job and tabu_entry[1] == j and tabu_entry[2] == i:
                        is_tabu = True
                        break
                
                # Aspiration: allow tabu move if it improves best
                if is_tabu and ms >= best_ms:
                    continue
                
                if ms < best_neighbor_ms:
                    best_neighbor_ms = ms
                    best_neighbor = new_order
                    best_neighbor_key = (job, i, j)
        
        if best_neighbor is None:
            # No improving non-tabu move found; try accepting any non-tabu move
            for i in range(n):
                job = best_order[i]
                for j in range(n):
                    if j == i:
                        continue
                    new_order = best_order[:i] + best_order[i+1:]
                    insert_pos = j if j < i else j
                    new_order = new_order[:insert_pos] + [job] + new_order[insert_pos:]
                    ms = _makespan_vectorized(processing_times, new_order)
                    
                    move_key = (job, i, j)
                    is_tabu = False
                    for tabu_entry in tabu_list:
                        if tabu_entry[0] == job and tabu_entry[1] == j and tabu_entry[2] == i:
                            is_tabu = True
                            break
                    
                    if not is_tabu:
                        best_neighbor = new_order
                        best_neighbor_ms = ms
                        best_neighbor_key = (job, i, j)
                        break
                if best_neighbor is not None:
                    break
        
        if best_neighbor is None:
            break  # Stuck, no moves available
        
        # Update tabu list
        tabu_list.append(best_neighbor_key)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        
        # Update current and best
        best_order = best_neighbor
        if best_neighbor_ms < best_ms:
            best_ms = best_neighbor_ms
        
        iteration += 1
    
    return list(best_order)
# EVOLVE-BLOCK-END
