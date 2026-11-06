"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _calc_makespan(processing_times, perm):
    """Calculate makespan for a given permutation using optimized Python lists."""
    n_machines = processing_times.shape[1]
    # perm is a list or array of job indices
    completion = [0.0] * n_machines
    
    for job in perm:
        # Access the row of processing times for this job
        row = processing_times[job]
        # Iterate over machines
        for m in range(n_machines):
            if m == 0:
                completion[0] += row[m]
            else:
                # completion[m] = max(completion[m], completion[m-1]) + row[m]
                c_prev = completion[m-1]
                c_curr = completion[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                completion[m] = c_curr + row[m]
    
    return completion[-1]


def _calc_makespan_incremental(processing_times, perm, start_pos):
    """
    Calculate makespan for perm, assuming completion times before start_pos are already known.
    Returns the final makespan.
    """
    n_machines = processing_times.shape[1]
    # We need the completion times at start_pos for all machines.
    # This function is intended to be used when we have the state from a previous prefix.
    # However, for simplicity in the current structure, we will stick to full calculation
    # but optimize the NEH step specifically.
    pass


def _neh_heuristic(processing_times):
    """
    NEH heuristic: 
    1. Sort jobs by total processing time descending.
    2. Insert each job into the best position in the current sequence.
    """
    n_jobs, n_machines = processing_times.shape
    
    # Sort by total processing time descending
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    # Start with the first job
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting at each position
        for pos in range(len(sequence) + 1):
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            
            # Optimized makespan calculation for insertion
            # We can compute the completion times up to pos-1, then insert job, then continue
            # But for now, let's just use the standard _calc_makespan
            # We can optimize this later if needed.
            ms = _calc_makespan(processing_times, new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence


def _local_search(processing_times, sequence, deadline):
    """
    Insertion local search: try moving a job to a different position.
    Use a time limit to stop before deadline.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all pairs of positions (i, j)
        # Move job at i to position j
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            job = current_seq[i]
            
            for j in range(n_jobs):
                if j == i:
                    continue
                if time.time() >= deadline:
                    break
                
                # Perform move
                seq_copy = current_seq[:]
                job_to_move = seq_copy.pop(i)
                if j > i:
                    insert_idx = j - 1
                else:
                    insert_idx = j
                
                seq_copy.insert(insert_idx, job_to_move)
                
                new_makespan = _calc_makespan(processing_times, seq_copy)
                
                if new_makespan < current_makespan:
                    current_seq = seq_copy
                    current_makespan = new_makespan
                    improved = True
                    break
            
            if improved:
                break
    
    return current_seq, current_makespan


def _simulated_annealing(processing_times, sequence, deadline):
    """
    Simulated annealing to escape local minima.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    # Temperature parameters
    T = current_makespan * 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Random perturbation: swap two jobs
        if n_jobs < 2:
            break
        i, j = random.sample(range(n_jobs), 2)
        new_seq = current_seq[:]
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        
        new_makespan = _calc_makespan(processing_times, new_seq)
        delta = new_makespan - current_makespan
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_seq = new_seq
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = list(current_seq)
        
        T *= cooling_rate
        if T < 1e-6:
            break
    
    return best_seq, best_makespan


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    # Step 1: NEH heuristic
    initial_seq = _neh_heuristic(processing_times)
    
    # Step 2: Local search with time limit
    best_seq, best_makespan = _local_search(processing_times, initial_seq, deadline)
    
    # Step 3: Simulated Annealing if time permits
    if time.time() < deadline - 5.0:
        sa_seq, sa_makespan = _simulated_annealing(processing_times, best_seq, deadline)
        if sa_makespan < best_makespan:
            best_seq = sa_seq
            best_makespan = sa_makespan
    
    # Step 4: Final local search polish
    if time.time() < deadline - 2.0:
        final_seq, final_makespan = _local_search(processing_times, best_seq, deadline)
        if final_makespan < best_makespan:
            best_seq = final_seq
    
    return best_seq


# EVOLVE-BLOCK-END
