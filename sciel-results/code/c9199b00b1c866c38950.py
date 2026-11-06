"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _calc_makespan(processing_times, perm):
    """
    Calculate makespan for a given permutation.
    processing_times: shape (n_jobs, n_machines)
    perm: list/array of job indices in order
    Returns: float makespan
    """
    n_machines = processing_times.shape[1]
    if len(perm) == 0:
        return 0.0
    
    # completion[m] = completion time of last job processed on machine m
    completion = np.zeros(n_machines, dtype=np.float64)
    
    for job_idx in perm:
        prev_completion = 0.0
        for m in range(n_machines):
            # Start time on machine m is max(prev completion on this machine, completion on previous machine for this job)
            start = max(completion[m], prev_completion)
            prev_completion = start + processing_times[job_idx, m]
        completion[-1] = prev_completion
        # Actually we need to update all machines
        # Let me redo this properly
    
    # Redo: proper DP
    completion = np.zeros(n_machines, dtype=np.float64)
    for job_idx in perm:
        new_completion = np.zeros(n_machines, dtype=np.float64)
        prev_c = 0.0
        for m in range(n_machines):
            start = max(completion[m], prev_c)
            prev_c = start + processing_times[job_idx, m]
            new_completion[m] = prev_c
        completion = new_completion
    
    return completion[-1]


def _calc_makespan_fast(processing_times, perm):
    """
    Faster makespan calculation using numpy where possible.
    """
    n_machines = processing_times.shape[1]
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    
    # Extract the submatrix for the permutation
    # P_perm[j, m] = processing_times[perm[j], m]
    P = processing_times[perm]  # shape (n_jobs, n_machines)
    
    # C[j, m] = completion time of job j on machine m
    # C[0, m] = C[0, m-1] + P[0, m]
    # C[j, 0] = C[j-1, 0] + P[j, 0]
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
    
    # For small n_jobs, a tight loop is fine
    # Let's use a simple iterative approach
    prev_row = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        curr_row = np.zeros(n_machines, dtype=np.float64)
        for m in range(n_machines):
            if m == 0:
                curr_row[m] = prev_row[m] + P[j, m]
            else:
                curr_row[m] = max(prev_row[m], curr_row[m-1]) + P[j, m]
        prev_row = curr_row
    
    return prev_row[-1]


def _neh_heuristic(processing_times):
    """
    NEH heuristic for flow shop scheduling.
    """
    n_jobs, n_machines = processing_times.shape
    
    # Step 1: Sort jobs in non-increasing order of total processing time
    total_pt = processing_times.sum(axis=1)
    order = list(np.argsort(-total_pt))
    
    # Step 2: Insert each job into the best position
    # For the first two jobs, try both orders and pick the one with smaller makespan
    if n_jobs <= 1:
        return order
    
    # Build the sequence incrementally
    # Start with the first job
    sequence = [order[0]]
    
    # For the second job, try both positions
    if n_jobs >= 2:
        j2 = order[1]
        # Option 1: [j1, j2]
        ms1 = _calc_makespan_fast(processing_times, [order[0], j2])
        # Option 2: [j2, j1]
        ms2 = _calc_makespan_fast(processing_times, [j2, order[0]])
        if ms1 <= ms2:
            sequence = [order[0], j2]
        else:
            sequence = [j2, order[0]]
    
    # Insert remaining jobs one by one into the best position
    for i in range(2, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        current_len = len(sequence)
        
        for pos in range(current_len + 1):
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            ms = _calc_makespan_fast(processing_times, new_seq)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        sequence = sequence[:best_pos] + [job] + sequence[best_pos:]
    
    return sequence


def _local_search(processing_times, initial_perm, max_iterations=500, time_limit=10.0):
    """
    Local search using 2-opt (swap) moves.
    """
    n_jobs = len(initial_perm)
    if n_jobs <= 1:
        return initial_perm
    
    best_perm = list(initial_perm)
    best_ms = _calc_makespan_fast(processing_times, best_perm)
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 50
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Try all swaps
        for i in range(n_jobs):
            for j in range(i+1, n_jobs):
                if i == j:
                    continue
                new_perm = list(best_perm)
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                ms = _calc_makespan_fast(processing_times, new_perm)
                if ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
    
    return best_perm


def _perturb(perm, strength=0.3):
    """
    Perturb a permutation by swapping some pairs or reversing a segment.
    """
    n = len(perm)
    if n <= 1:
        return list(perm)
    
    new_perm = list(perm)
    n_perturbations = max(1, int(strength * n))
    
    for _ in range(n_perturbations):
        choice = random.random()
        if choice < 0.5:
            # Swap two random positions
            i, j = random.sample(range(n), 2)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        else:
            # Reverse a random segment
            i, j = sorted(random.sample(range(n), 2))
            new_perm[i:j+1] = new_perm[i:j+1][::-1]
    
    return new_perm


def order_jobs(processing_times):
    """
    Main entry point. Constructs a job permutation minimizing makespan.
    Uses NEH heuristic followed by local search with restarts.
    """
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    global_start = time.time()
    deadline = 140.0  # Leave 10s margin before 150s kill
    
    # First, get a good initial solution using NEH
    neh_perm = _neh_heuristic(processing_times)
    
    # Local search from NEH
    best_perm = _local_search(processing_times, neh_perm, max_iterations=1000, time_limit=20.0)
    best_ms = _calc_makespan_fast(processing_times, best_perm)
    
    # Check time
    elapsed = time.time() - global_start
    if elapsed > deadline - 5:
        return list(best_perm)
    
    # Multi-start local search with perturbation
    n_restarts = 0
    max_restarts = 50
    
    while True:
        elapsed = time.time() - global_start
        if elapsed > deadline - 5:
            break
        if n_restarts >= max_restarts:
            break
        
        # Perturb the best solution
        perturbed = _perturb(best_perm, strength=0.2 + 0.1 * random.random())
        
        # Local search from perturbed solution
        candidate = _local_search(processing_times, perturbed, max_iterations=300, time_limit=5.0)
        candidate_ms = _calc_makespan_fast(processing_times, candidate)
        
        if candidate_ms < best_ms - 1e-9:
            best_perm = candidate
            best_ms = candidate_ms
        
        n_restarts += 1
    
    return list(best_perm)
# EVOLVE-BLOCK-END
