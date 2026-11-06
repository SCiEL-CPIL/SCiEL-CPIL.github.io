"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _makespan(order, P):
    """
    Calculate the makespan for a given job order in a flow shop.
    P: (n_jobs, n_machines) array
    order: list of job indices
    """
    if not order:
        return 0.0
    
    n_machines = P.shape[1]
    if n_machines == 0:
        return 0.0
    
    # C[j] is the completion time of the current job on machine j
    # We track the completion time of the previous job on each machine
    prev_completion = np.zeros(n_machines)
    
    for j in order:
        # Processing time for job j on all machines
        p_j = P[j]
        
        # Completion time on machine 0
        curr_completion = prev_completion[0] + p_j[0]
        
        if n_machines == 1:
            prev_completion[0] = curr_completion
            continue
            
        # For subsequent machines
        for m in range(1, n_machines):
            curr_completion = max(curr_completion, prev_completion[m]) + p_j[m]
            prev_completion[m] = curr_completion
            
    return prev_completion[-1]

def _eval_insertion(sequence, job, pos, P):
    """
    Evaluate the makespan of inserting 'job' at position 'pos' in 'sequence'.
    Optimized: computes prefix completions once, then evaluates all insertion positions.
    """
    n_machines = P.shape[1]
    if n_machines == 0:
        return 0.0
    
    n = len(sequence)
    
    # Precompute prefix completions: prefix[k][m] = completion time after processing sequence[0..k-1] on machine m
    # prefix[0] = zeros
    # prefix[k+1] = completion after sequence[0..k]
    prefix = np.zeros((n + 1, n_machines))
    
    for k in range(n):
        j = sequence[k]
        p_j = P[j]
        curr = prefix[k][0] + p_j[0]
        prefix[k + 1, 0] = curr
        for m in range(1, n_machines):
            curr = max(curr, prefix[k][m]) + p_j[m]
            prefix[k + 1, m] = curr
    
    # Now evaluate insertion at position pos:
    # Jobs before pos: sequence[0..pos-1], completion state = prefix[pos]
    # Insert job, then process sequence[pos..n-1]
    
    state = prefix[pos].copy()
    p_job = P[job]
    curr = state[0] + p_job[0]
    for m in range(1, n_machines):
        curr = max(curr, state[m]) + p_job[m]
        state[m] = curr
    
    # Process remaining jobs
    for k in range(pos, n):
        j = sequence[k]
        p_j = P[j]
        curr = state[0] + p_j[0]
        for m in range(1, n_machines):
            curr = max(curr, state[m]) + p_j[m]
            state[m] = curr
    
    return state[-1]

def _neh_heuristic(P):
    """
    Nawaz, Enscore, and Ham (NEH) heuristic.
    """
    n_jobs, n_machines = P.shape
    
    # Sum of processing times for each job
    sums = P.sum(axis=1)
    
    # Sort jobs by total processing time in descending order
    sorted_jobs = list(np.argsort(-sums))
    
    sequence = []
    
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        
        # Try inserting the job at every position
        for pos in range(len(sequence) + 1):
            ms = _eval_insertion(sequence, job, pos, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
        
    return sequence

def _local_search(initial_order, P, time_limit=110.0):
    """
    Local search using insertion moves with optimized evaluation.
    """
    start_time = time.time()
    order = list(initial_order)
    current_ms = _makespan(order, P)
    n_jobs = len(order)
    
    if n_jobs <= 1:
        return order, current_ms
    
    improved = True
    while improved:
        improved = False
        # Check time limit
        if time.time() - start_time > time_limit:
            break
            
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            job = order[i]
            # Try moving job i to every position j != i
            best_new_ms = current_ms
            best_insert_idx = -1
            best_j = -1
            
            # Precompute prefix completions for the sequence without job i
            seq_without = order[:i] + order[i+1:]
            n_without = len(seq_without)
            
            # Compute prefix completions for seq_without
            n_machines = P.shape[1]
            prefix = np.zeros((n_without + 1, n_machines))
            for k in range(n_without):
                j = seq_without[k]
                p_j = P[j]
                curr = prefix[k][0] + p_j[0]
                prefix[k + 1, 0] = curr
                for m in range(1, n_machines):
                    curr = max(curr, prefix[k][m]) + p_j[m]
                    prefix[k + 1, m] = curr
            
            # Try all insertion positions
            for insert_idx in range(n_without + 1):
                if time.time() - start_time > time_limit:
                    break
                
                # Evaluate insertion at insert_idx using prefix
                state = prefix[insert_idx].copy()
                p_job = P[job]
                curr = state[0] + p_job[0]
                for m in range(1, n_machines):
                    curr = max(curr, state[m]) + p_job[m]
                    state[m] = curr
                
                # Process remaining jobs
                for k in range(insert_idx, n_without):
                    j = seq_without[k]
                    p_j = P[j]
                    curr = state[0] + p_j[0]
                    for m in range(1, n_machines):
                        curr = max(curr, state[m]) + p_j[m]
                        state[m] = curr
                
                new_ms = state[-1]
                
                if new_ms < best_new_ms:
                    best_new_ms = new_ms
                    best_insert_idx = insert_idx
                    best_j = insert_idx  # track for later
            
            if best_insert_idx >= 0 and best_new_ms < current_ms:
                order = seq_without
                order.insert(best_insert_idx, job)
                current_ms = best_new_ms
                improved = True
                break
            if time.time() - start_time > time_limit:
                break
                
    return order, current_ms

def _simulated_annealing(initial_order, P, time_limit=110.0):
    """
    Simulated annealing to escape local optima found by local search.
    Uses insertion moves instead of swaps for better neighborhood coverage.
    """
    start_time = time.time()
    order = list(initial_order)
    current_ms = _makespan(order, P)
    n_jobs = len(order)
    
    if n_jobs <= 1:
        return order, current_ms
    
    best_order = list(order)
    best_ms = current_ms
    
    # Initial temperature
    T = 1.0
    cooling_rate = 0.995
    
    while time.time() - start_time < time_limit:
        # Generate a random insertion move
        i = np.random.randint(n_jobs)
        j = np.random.randint(n_jobs)
        if i == j:
            continue
        
        # Remove job at i and insert at j
        job = order[i]
        seq_without = order[:i] + order[i+1:]
        insert_idx = j if j <= i else j - 1
        
        new_order = seq_without
        new_order.insert(insert_idx, job)
        
        new_ms = _makespan(new_order, P)
        
        delta = new_ms - current_ms
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or (T > 1e-10 and np.random.rand() < np.exp(-delta / T)):
            order = new_order
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = list(order)
        
        T *= cooling_rate
        
        # Check time limit
        if time.time() - start_time > time_limit:
            break
            
    return best_order, best_ms

def order_jobs(processing_times):
    # Convert to numpy array if not already
    P = np.asarray(processing_times)
    
    if P.size == 0:
        return []
    
    n_jobs = P.shape[0]
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # 1. Generate initial solution using NEH
    best_order = _neh_heuristic(P)
    best_ms = _makespan(best_order, P)
    
    # 2. Improve using Local Search (first pass)
    # We have a 150s limit. Reserve 5s for safety.
    improved_order, improved_ms = _local_search(best_order, P, time_limit=120.0)
    
    if improved_ms < best_ms:
        best_order = improved_order
        best_ms = improved_ms
    
    # 3. Simulated Annealing to escape local optima
    sa_order, sa_ms = _simulated_annealing(best_order, P, time_limit=20.0)
    
    if sa_ms < best_ms:
        best_order = sa_order
        best_ms = sa_ms
    
    # 4. Final polish: run local search again on the best solution
    final_order, final_ms = _local_search(best_order, P, time_limit=3.0)
    
    if final_ms < best_ms:
        best_order = final_order
        best_ms = final_ms
    
    return best_order
# EVOLVE-BLOCK-END
