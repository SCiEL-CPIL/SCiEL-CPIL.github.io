"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

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
    prev = np.zeros(n_machines)
    
    for job in order:
        p = processing_times[job]
        # Machine 0
        curr_0 = prev[0] + p[0]
        # Machines 1..M-1
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

def _local_search_first_improvement(seq, processing_times, max_iterations=1000):
    """
    First-improvement local search using insertion moves.
    Returns the improved sequence and its makespan.
    """
    n_jobs = len(seq)
    current_seq = seq[:]
    current_ms = _makespan(current_seq, processing_times)
    
    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        for i in range(n_jobs):
            for j in range(n_jobs):
                if i == j:
                    continue
                # Skip if adjacent and same direction (no-op)
                if j == i - 1 or j == i + 1:
                    continue
                    
                # Remove job at i
                job = current_seq[i]
                temp_seq = current_seq[:i] + current_seq[i+1:]
                
                # Insert at j
                # If j > i, the index shifts left by 1 in temp_seq
                if j > i:
                    j_idx = j - 1
                else:
                    j_idx = j
                    
                candidate_seq = temp_seq[:j_idx] + [job] + temp_seq[j_idx:]
                
                candidate_ms = _makespan(candidate_seq, processing_times)
                
                if candidate_ms < current_ms:
                    current_seq = candidate_seq
                    current_ms = candidate_ms
                    improved = True
                    break
            if improved:
                break
                
    return current_seq, current_ms

def _iterated_local_search(processing_times, time_limit=115.0):
    """
    Iterated Local Search (ILS) for flow-shop scheduling.
    Uses NEH for initial solution, then perturbation and local search.
    """
    n_jobs, n_machines = processing_times.shape
    start_time = time.time()
    
    # Initial solution using NEH
    best_seq = _neh_initial(processing_times)
    best_ms = _makespan(best_seq, processing_times)
    
    # Local search on initial solution
    best_seq, best_ms = _local_search_first_improvement(best_seq, processing_times)
    
    # ILS loop
    perturbation_strength = 2
    while time.time() - start_time < time_limit:
        # Perturb the best solution
        current_seq = best_seq[:]
        n = len(current_seq)
        
        # Apply 'strength' random insertion moves
        for _ in range(perturbation_strength):
            if n < 2:
                break
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            job = current_seq[i]
            current_seq.pop(i)
            if j > i:
                j -= 1
            current_seq.insert(j, job)
        
        # Local search on perturbed solution
        candidate_seq, candidate_ms = _local_search_first_improvement(current_seq, processing_times)
        
        # Acceptance criterion: accept if better
        if candidate_ms < best_ms:
            best_ms = candidate_ms
            best_seq = candidate_seq[:]
            
        # Occasionally increase perturbation strength to escape deep local optima
        if np.random.rand() < 0.1:
            perturbation_strength = np.random.randint(3, 6)
        else:
            perturbation_strength = 2
            
    return best_seq, best_ms

def order_jobs(processing_times):
    """
    Main entry point. Constructs a job permutation minimizing makespan.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Use ILS with a time limit
    best_seq, best_ms = _iterated_local_search(processing_times, time_limit=115.0)
    
    return best_seq
# EVOLVE-BLOCK-END
