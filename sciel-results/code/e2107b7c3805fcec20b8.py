"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using a multi-stage heuristic:
    1. NEH (Nawaz, Enscore, Ham) heuristic for initial construction.
    2. Local search with insertion moves (I1) to refine the solution.
    
    The NEH heuristic is known to perform well on flow-shop problems.
    Local search with insertion moves further improves the solution.
    """
    n_jobs, n_machines = processing_times.shape
    
    # --- Stage 1: NEH Heuristic ---
    # Sort jobs by decreasing total processing time
    total_times = processing_times.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    # Build the sequence using NEH rule
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting job j at each possible position
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [j] + seq[pos:]
            # Calculate makespan for this candidate
            makespan = _calc_makespan(processing_times, candidate, n_machines)
            if makespan < best_makespan:
                best_makespan = makespan
                best_pos = pos
        
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    
    # --- Stage 2: Local Search with Insertion Moves ---
    # Try to improve by moving each job to a different position
    improved = True
    max_iterations = 50  # Limit iterations to avoid timeout
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(n_jobs):
            job = seq[i]
            # Try inserting job at each other position
            for j in range(n_jobs):
                if i == j:
                    continue
                # Remove job from position i
                candidate = seq[:i] + seq[i+1:]
                # Adjust insertion position if j > i
                insert_pos = j if j < i else j - 1
                # Insert job at new position
                candidate = candidate[:insert_pos] + [job] + candidate[insert_pos:]
                
                # Calculate makespan for this candidate
                new_makespan = _calc_makespan(processing_times, candidate, n_machines)
                current_makespan = _calc_makespan(processing_times, seq, n_machines)
                
                if new_makespan < current_makespan:
                    seq = candidate
                    improved = True
                    break
            if improved:
                break
    
    return seq


def _calc_makespan(processing_times, seq, n_machines):
    """Calculate the makespan for a given job sequence."""
    # completion_times[m] is the completion time on machine m for the current job
    completion_times = np.zeros(n_machines)
    
    for job_idx in seq:
        for m in range(n_machines):
            if m == 0:
                completion_times[m] += processing_times[job_idx, m]
            else:
                completion_times[m] = max(completion_times[m], completion_times[m-1]) + processing_times[job_idx, m]
    
    return completion_times[-1]
# EVOLVE-BLOCK-END
