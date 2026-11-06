# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _calc_makespan(processing_times, perm):
    """
    Calculate makespan for a given permutation efficiently.
    processing_times: (n_jobs, n_machines) array
    perm: list or array of job indices in order
    Returns the makespan (float).
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return 0.0
    
    # completion[m] is the completion time of the last job processed on machine m
    completion = np.zeros(n_machines, dtype=np.float64)
    
    for job_idx in perm:
        # For each machine, the completion time is max(prev_machine_completion, current_machine_completion) + processing_time
        # We need to iterate through machines sequentially because of the dependency
        prev_completion = 0.0
        for m in range(n_machines):
            pt = processing_times[job_idx, m]
            # The job can start on machine m only after it finishes on machine m-1 (prev_completion)
            # AND machine m is free (completion[m])
            start_time = max(prev_completion, completion[m])
            prev_completion = start_time + pt
            completion[m] = prev_completion
        
    return completion[-1]


def _neh_heuristic(processing_times):
    """
    NEH (Nawaz-Enscore-Ham) heuristic for flow shop scheduling.
    Returns a permutation of job indices.
    """
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    # Build the sequence one job at a time
    seq = [sorted_jobs[0]]
    
    for i in range(1, n_jobs):
        job = sorted_jobs[i]
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting job at each position in the current sequence
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            # Calculate makespan for this candidate sequence
            # For speed, we can stop early if it's already worse, but for simplicity we calculate fully
            # Optimization: Only check a subset of positions if n is large, but for now check all
            ms = _calc_makespan(processing_times, new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        seq.insert(best_pos, job)
    
    return seq


def _two_opt_local_search(processing_times, initial_perm, max_iterations=1000, time_limit=10.0):
    """
    2-opt local search to improve the makespan.
    Tries swapping two jobs in the permutation.
    """
    n_jobs = len(initial_perm)
    if n_jobs < 2:
        return initial_perm
    
    current_perm = list(initial_perm)
    current_ms = _calc_makespan(processing_times, current_perm)
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 50  # Stop if no improvement for this many iterations
    
    for iteration in range(max_iterations):
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Randomly select pairs to swap to avoid full O(n^2) scan every time, 
        # or do a full scan if n is small.
        # For n < 100, full scan is fine. For larger n, random sampling is better.
        if n_jobs <= 100:
            pairs_to_check = [(i, j) for i in range(n_jobs) for j in range(i + 1, n_jobs)]
            random.shuffle(pairs_to_check)
        else:
            # Sample a fixed number of random pairs
            num_samples = min(200, n_jobs * (n_jobs - 1) // 2)
            pairs_to_check = []
            for _ in range(num_samples):
                i = random.randint(0, n_jobs - 1)
                j = random.randint(0, n_jobs - 1)
                if i != j:
                    if i > j:
                        i, j = j, i
                    pairs_to_check.append((i, j))
        
        for i, j in pairs_to_check:
            if i == j:
                continue
            
            # Swap jobs at positions i and j
            new_perm = current_perm.copy()
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            
            new_ms = _calc_makespan(processing_times, new_perm)
            
            if new_ms < current_ms - 1e-9:
                current_perm = new_perm
                current_ms = new_ms
                improved = True
                no_improve_count = 0
                break  # Restart the loop with the new permutation
            else:
                # Undo swap (not needed since we used a copy, but good practice if modifying in place)
                pass
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
    
    return current_perm


def order_jobs(processing_times):
    """
    Main entry point.
    Uses NEH heuristic followed by 2-opt local search.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin for the 150s limit
    
    n_jobs = processing_times.shape[0]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Construct initial solution using NEH
    best_perm = _neh_heuristic(processing_times)
    best_ms = _calc_makespan(processing_times, best_perm)
    
    # 2. Improve using 2-opt local search
    # We can run multiple restarts if time permits
    num_restarts = 0
    max_restarts = 5  # Limit restarts to avoid infinite loops
    
    while time.time() < deadline:
        # Check if we have enough time for another local search pass
        # Estimate time for local search. If n is large, it might take a while.
        # For now, just run it and let the internal time limit handle it.
        
        # Randomly perturb the best solution for a restart (if not the first)
        if num_restarts > 0:
            # Simple perturbation: swap two random jobs
            perturbed_perm = best_perm.copy()
            i = random.randint(0, n_jobs - 1)
            j = random.randint(0, n_jobs - 1)
            if i != j:
                perturbed_perm[i], perturbed_perm[j] = perturbed_perm[j], perturbed_perm[i]
        else:
            perturbed_perm = best_perm
        
        # Run local search
        local_search_time = min(10.0, max(1.0, deadline - (time.time() - start_time)))
        improved_perm = _two_opt_local_search(processing_times, perturbed_perm, 
                                               max_iterations=1000, 
                                               time_limit=local_search_time)
        
        improved_ms = _calc_makespan(processing_times, improved_perm)
        
        if improved_ms < best_ms - 1e-9:
            best_perm = improved_perm
            best_ms = improved_ms
        
        num_restarts += 1
        
        # If we've done a few restarts, break to ensure we return in time
        if num_restarts >= max_restarts:
            break
    
    return list(best_perm)
# EVOLVE-BLOCK-END
