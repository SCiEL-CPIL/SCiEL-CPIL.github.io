"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def _makespan(order, processing_times):
    """
    Fast makespan calculation.
    order: list of job indices
    processing_times: np.ndarray [n_jobs, n_machines]
    """
    n_machines = processing_times.shape[1]
    if len(order) == 0:
        return 0.0
    
    # prev[m] is completion time of previous job on machine m
    # curr[m] is completion time of current job on machine m
    prev = np.zeros(n_machines)
    
    for job in order:
        p = processing_times[job]
        # Machine 0
        curr_0 = prev[0] + p[0]
        # Machines 1..M-1
        # curr[m] = max(curr[m-1], prev[m]) + p[m]
        # We can do this iteratively
        curr_m_minus_1 = curr_0
        for m in range(1, n_machines):
            curr_m = max(curr_m_minus_1, prev[m]) + p[m]
            prev[m] = curr_m
            curr_m_minus_1 = curr_m
        prev[0] = curr_0
        
    return prev[-1]

def _neh_initial(processing_times):
    """Generate initial solution using NEH heuristic."""
    n_jobs, n_machines = processing_times.shape
    total_times = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_times)
    
    sequence = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        sequence.insert(best_pos, job)
    return sequence

def _simulated_annealing(initial_seq, processing_times, max_iters=5000, t_start=100.0, t_end=1.0, cooling_rate=0.999):
    """
    Run Simulated Annealing starting from initial_seq.
    Uses insertion moves.
    """
    n_jobs = len(initial_seq)
    current_seq = initial_seq[:]
    current_ms = _makespan(current_seq, processing_times)
    best_seq = current_seq[:]
    best_ms = current_ms
    
    t = t_start
    i = 0
    while t > t_end and i < max_iters:
        # Generate neighbor: Insertion Move
        # Pick two distinct indices i, j
        if n_jobs < 2:
            break
            
        i_idx = np.random.randint(n_jobs)
        j_idx = np.random.randint(n_jobs)
        
        if i_idx == j_idx:
            continue
            
        # Remove job at i_idx
        job = current_seq[i_idx]
        temp_seq = current_seq[:i_idx] + current_seq[i_idx+1:]
        
        # Insert at j_idx
        # If j_idx > i_idx, the index shifts left by 1 in temp_seq
        if j_idx > i_idx:
            j_idx -= 1
            
        candidate_seq = temp_seq[:j_idx] + [job] + temp_seq[j_idx:]
        
        candidate_ms = _makespan(candidate_seq, processing_times)
        delta = candidate_ms - current_ms
        
        if delta < 0:
            # Accept improvement
            current_seq = candidate_seq
            current_ms = candidate_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_seq = current_seq[:]
        else:
            # Accept with probability exp(-delta / t)
            if np.random.rand() < np.exp(-delta / t):
                current_seq = candidate_seq
                current_ms = candidate_ms
                
        t *= cooling_rate
        i += 1
        
    return best_seq, best_ms

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    # If very small, exact search? No, SA is fine.
    # Multi-start SA
    best_global_seq = []
    best_global_ms = float('inf')
    
    # Number of restarts depends on size. 
    # For small N, more restarts. For large N, fewer but longer runs.
    if n_jobs <= 10:
        restarts = 20
        max_iters = 2000
    elif n_jobs <= 20:
        restarts = 10
        max_iters = 5000
    else:
        restarts = 5
        max_iters = 10000
        
    # 1. Start with NEH
    seq_neh = _neh_initial(processing_times)
    best_seq, best_ms = _simulated_annealing(seq_neh, processing_times, max_iters=max_iters)
    if best_ms < best_global_ms:
        best_global_ms = best_ms
        best_global_seq = best_seq
        
    # 2. Random starts
    for _ in range(restarts - 1):
        seq_random = list(np.random.permutation(n_jobs))
        best_seq, best_ms = _simulated_annealing(seq_random, processing_times, max_iters=max_iters)
        if best_ms < best_global_ms:
            best_global_ms = best_ms
            best_global_seq = best_seq
            
    return best_global_seq
# EVOLVE-BLOCK-END
