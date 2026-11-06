"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan_from_order(order, processing_times):
    n_machines = processing_times.shape[1]
    if n_machines == 0 or len(order) == 0:
        return 0.0
    completion = np.zeros(n_machines)
    for job in order:
        p = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            prev = max(prev, completion[m]) + p[m]
            completion[m] = prev
    return float(completion[-1])


def _compute_makespan_vectorized(order, processing_times):
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    seq = np.asarray(order, dtype=np.int64)
    P = processing_times[seq]
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
    # Use a small Python loop over jobs but vectorize over machines where possible.
    prev_row = np.zeros(n_machines, dtype=float)
    for j in range(len(seq)):
        p = P[j]
        # For each machine, C_jm = max(prev_row[m], C_j,m-1) + p[m]
        # This dependency is sequential over m, but n_machines is often small.
        cur = np.empty(n_machines, dtype=float)
        left = 0.0
        for m in range(n_machines):
            left = max(left, prev_row[m]) + p[m]
            cur[m] = left
        prev_row = cur
    return float(prev_row[-1])


def _neh_initialization(processing_times, rng=None):
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    if rng is None:
        rng = np.random.default_rng()

    # NEH-like: sort by total processing time, but add small randomization to get diverse starts.
    totals = processing_times.sum(axis=1)
    # Add small noise to break ties and create diversity.
    noise = rng.random(n_jobs) * 0.01 * (np.max(totals) - np.min(totals) + 1e-9)
    order_keys = - (totals + noise)
    initial_order = list(np.argsort(order_keys))

    # Standard NEH insertion: build sequence by inserting each job at best position.
    seq = []
    for job in initial_order:
        best_pos = 0
        best_ms = float('inf')
        # Try all insertion positions
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [job] + seq[pos:]
            ms = _makespan_from_order(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    return seq


def _improve_with_local_search(order, processing_times, max_time=None, rng=None):
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return order

    if rng is None:
        rng = np.random.default_rng()

    current = list(order)
    current_ms = _makespan_from_order(current, processing_times)
    best = list(current)
    best_ms = current_ms

    start_time = time.time()
    deadline = start_time + (max_time if max_time is not None else 10.0)

    # Fast incremental evaluation for insertion moves.
    # We precompute completion matrix for current sequence.
    def compute_completion_matrix(seq):
        n_m = processing_times.shape[1]
        C = np.zeros((len(seq), n_m), dtype=float)
        if len(seq) == 0:
            return C
        for j in range(len(seq)):
            job = seq[j]
            p = processing_times[job]
            left = 0.0
            for m in range(n_m):
                if j > 0:
                    left = max(left, C[j-1][m])
                else:
                    left = max(left, 0.0)
                left += p[m]
                C[j][m] = left
        return C

    C = compute_completion_matrix(current)
    n = len(current)

    # Helper: evaluate insertion of job at position i to position j (i < j means move right, i > j means move left)
    # We'll use a simplified incremental approach: recompute only affected part.
    # For speed, we'll use a different strategy: for each job, try all positions, but use a faster makespan calc.
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 0.5:
            break
            
        # Try moving each job to a different position
        for i in range(n):
            if time.time() > deadline - 0.5:
                break
            job = current[i]
            # Try all possible positions
            for j in range(n):
                if i == j:
                    continue
                # Skip if we're just moving by one position (will be tried elsewhere)
                # Actually, let's try all positions
                if time.time() > deadline - 0.5:
                    break
                    
                # Create new sequence
                if i < j:
                    new_seq = current[:i] + current[i+1:j+1] + [job] + current[j+1:]
                else:
                    new_seq = current[:j] + [job] + current[j:i] + current[i+1:]
                    
                ms = _makespan_from_order(new_seq, processing_times)
                if ms < current_ms - 1e-9:
                    current = new_seq
                    current_ms = ms
                    C = compute_completion_matrix(current)
                    improved = True
                    if current_ms < best_ms - 1e-9:
                        best = list(current)
                        best_ms = current_ms
                    break  # Restart i loop since sequence changed
            if improved:
                break  # Restart from beginning
                
    return best


def order_jobs(processing_times):
    start_time = time.time()
    # Use 120s limit, but leave 10s margin for safety
    time_budget = 110.0
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    rng = np.random.default_rng()
    
    # Phase 1: Generate diverse initial solutions using NEH with randomization
    # Number of initial solutions depends on time budget
    n_initial = max(3, min(20, int(time_budget / 5)))  # 3-20 initial solutions
    
    best_order = None
    best_ms = float('inf')
    
    for i in range(n_initial):
        if time.time() - start_time > time_budget * 0.3:
            break
            
        # Generate NEH solution with randomization
        order = _neh_initialization(processing_times, rng)
        ms = _makespan_from_order(order, processing_times)
        
        if ms < best_ms:
            best_ms = ms
            best_order = list(order)
    
    if best_order is None:
        # Fallback: simple LPT
        return list(np.argsort(-processing_times.sum(axis=1)))
    
    # Phase 2: Local search improvement with time budget
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 5:
        best_order = _improve_with_local_search(best_order, processing_times, max_time=remaining_time, rng=rng)
        best_ms = _makespan_from_order(best_order, processing_times)
    
    # Phase 3: If time remains, try more initial solutions and improve
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 10:
        # Try a few more diverse starts
        n_more = max(2, min(10, int(remaining_time / 5)))
        for i in range(n_more):
            if time.time() - start_time > time_budget - 5:
                break
                
            order = _neh_initialization(processing_times, rng)
            ms = _makespan_from_order(order, processing_times)
            
            if ms < best_ms:
                best_ms = ms
                best_order = list(order)
                
                # Quick local search if time permits
                remaining = time_budget - (time.time() - start_time)
                if remaining > 5:
                    best_order = _improve_with_local_search(best_order, processing_times, max_time=remaining, rng=rng)
                    best_ms = _makespan_from_order(best_order, processing_times)
    
    return best_order
# EVOLVE-BLOCK-END
