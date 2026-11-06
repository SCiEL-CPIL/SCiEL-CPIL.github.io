"""
Iterated Local Search (ILS) for Flow Shop Scheduling.
Strategy:
1. Initialize with NEH heuristic.
2. Intensive Local Search using Insertion Moves with O(n*m) delta evaluation.
3. Iterated Perturbation: Pick a random job and insert it at a random new position (Shake).
4. Accept if better, or with probability exp(-delta/T) (Simulated Annealing style acceptance for diversification).
5. Repeat until time limit.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def _makespan(perm, processing_times):
    """
    Compute makespan efficiently.
    perm: list of job indices
    processing_times: np.ndarray [n_jobs, n_machines]
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    # C[j, m] completion time of job j on machine m
    # Recurrence: C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
    # We can do this in-place row by row or column by column.
    # For speed, we process job by job, updating a running array of machine times.
    
    machine_times = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            # Machine m is busy until machine_times[m]
            # Previous machine (m-1) finished job at prev
            # Job must start after both are free
            start_time = max(machine_times[m], prev)
            end_time = start_time + pt[m]
            machine_times[m] = end_time
            prev = end_time
            
    return machine_times[-1]

def _delta_insertion(perm, processing_times, i, j):
    """
    Calculate the change in makespan if we remove job at position i and insert it at position j.
    If i < j, the job moves from i to j.
    If i > j, the job moves from i to j.
    
    This is O(n*m) in the worst case if we recompute, but we can do better by 
    recomputing the makespan of the affected segment. However, for simplicity and robustness 
    in Python, a full recomputation of the makespan for the candidate is O(n*m). 
    Given n is likely small (e.g., < 50) and m is small (< 20), O(n*m) is ~1000 ops.
    150s / 1000 ops = 150,000 iterations. This is feasible.
    
    To be faster, we can implement a proper delta, but let's start with a robust 
    "remove and insert" makespan check which is often sufficient if n is small.
    If n is large, we need the O(n*m) delta for the *segment*.
    
    Let's implement a slightly optimized makespan that breaks early if possible? No, makespan is global.
    
    Let's stick to a very tight loop for makespan.
    """
    # Create candidate permutation
    # To avoid list copying overhead in the hot loop, we can pass the list and modify it, 
    # but that's risky. Let's create the candidate list.
    # Actually, we can compute the makespan of the candidate directly without creating the list 
    # if we write a specialized makespan function that takes indices.
    
    # Let's write a specialized makespan that takes perm and indices i, j to simulate the move.
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    # Simulate the permutation
    # We need to iterate through the "new" order.
    # New order: perm[0..j-1] + perm[i] + perm[j..i-1] + perm[i+1..]  (if i < j)
    # Or: perm[0..j-1] + perm[i] + perm[j+1..i] + perm[i+1..] (if i > j)
    
    # This is complex to do in one pass without a list. 
    # Given the constraints (120s), and typical n (let's assume n < 100), 
    # creating a list and calling _makespan is likely fine if _makespan is fast.
    
    candidate = list(perm)
    job_to_move = candidate.pop(i)
    candidate.insert(j, job_to_move)
    
    # Compute makespan of candidate
    machine_times = np.zeros(n_machines, dtype=np.float64)
    for j_idx in range(n_jobs):
        job = candidate[j_idx]
        pt = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            start_time = max(machine_times[m], prev)
            end_time = start_time + pt[m]
            machine_times[m] = end_time
            prev = end_time
            
    return machine_times[-1]

def _local_search_insertion(perm, processing_times, time_limit):
    """
    Local search using insertion moves.
    Iterates through all possible (i, j) pairs.
    Accepts the first improving move (steepest ascent? No, first improvement is faster).
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm, _makespan(perm, processing_times)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try all insertion moves
        # Optimization: Only try i < j and i > j? 
        # Moving i to j is the same as moving j to i in terms of the set of permutations, 
        # but the direction matters for the "first improvement" strategy.
        
        for i in range(n_jobs):
            if (time.time() - start_time) >= time_limit:
                break
            for j in range(n_jobs):
                if i == j or abs(i - j) == 1:
                    continue
                
                # Evaluate delta
                new_ms = _delta_insertion(best_perm, processing_times, i, j)
                
                if new_ms < best_ms - 1e-9:
                    # Accept move
                    job_to_move = best_perm.pop(i)
                    best_perm.insert(j, job_to_move)
                    best_ms = new_ms
                    improved = True
                    break # Restart the loop with the new permutation
            if improved:
                break
                
    return best_perm, best_ms

def _neh_order(processing_times):
    """
    NEH heuristic.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0: return []
    if n_jobs == 1: return [0]
    
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
        
    return perm

def order_jobs(processing_times):
    """
    Main entry point.
    Uses NEH to initialize, then Iterated Local Search with Insertion moves.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Global deadline with safety margin
    global_deadline = time.time() + 140.0
    
    # 1. Initialize with NEH
    current_perm = _neh_order(processing_times)
    current_ms = _makespan(current_perm, processing_times)
    
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # 2. Initial Local Search to get to a local optimum
    # Allocate some time for initial polish
    initial_ls_time = min(10.0, (global_deadline - time.time()) * 0.1)
    current_perm, current_ms = _local_search_insertion(current_perm, processing_times, initial_ls_time)
    
    if current_ms < best_ms:
        best_perm = list(current_perm)
        best_ms = current_ms
    
    # 3. Iterated Local Search
    # Temperature for acceptance
    T = 1.0
    T_start = 50.0
    T_end = 1.0
    cooling_rate = 0.999 # Very slow cooling, or just fixed T? 
    # Let's use a simple geometric cooling based on iterations or time.
    # Since we don't know the number of iterations in advance, let's use time-based or just a fixed T 
    # that decays slowly. Or better, use a "restart" strategy: if no improvement for X iterations, 
    # increase perturbation strength.
    
    # Let's stick to a simple SA-like acceptance with a fixed T that decays over time.
    # Or even simpler: Accept if better, or with prob 0.05 if worse (random walk).
    # ILS usually accepts only improvements, but the "Shake" provides the diversification.
    # If we accept worse solutions, it's more like SA. 
    # Standard ILS: 
    # 1. Local Search
    # 2. Perturb
    # 3. Local Search
    # 4. Accept if better than Best, or with probability p.
    
    # Let's use: Accept if better than Best. Otherwise, keep Current.
    # This is "Best-First" ILS. It can get stuck.
    # Let's use "Accept if better than Current" (Standard ILS).
    
    start_time = time.time()
    iteration = 0
    
    while (time.time() - start_time) < (global_deadline - time.time()) * 0.95:
        # Check global deadline
        if time.time() > global_deadline - 5.0:
            break
            
        # Perturbation: Move a job to a new position
        # Strength of perturbation can vary. 
        # Let's pick two random positions i, j.
        # If we just pick i, j randomly, the "distance" is random.
        # To ensure we escape local optima, we might want to move a job far away.
        
        i = random.randint(0, n_jobs - 1)
        j = random.randint(0, n_jobs - 1)
        
        if i == j:
            continue
            
        # Perform move
        candidate_perm = list(current_perm)
        job_to_move = candidate_perm.pop(i)
        candidate_perm.insert(j, job_to_move)
        
        # Local Search on the perturbed solution
        # Allocate a small time budget for this local search
        # We need to be careful not to spend too much time on a bad perturbation.
        # Let's limit the local search to a small number of iterations or time.
        # Since _local_search_insertion runs until no improvement, it might take a while.
        # We should probably limit the number of "passes" in local search for the ILS loop.
        
        # Modified Local Search for ILS: Limit to a few passes or time slice.
        ls_time_limit = min(2.0, (global_deadline - time.time()) * 0.1)
        perturbed_perm, perturbed_ms = _local_search_insertion(candidate_perm, processing_times, ls_time_limit)
        
        # Acceptance Criterion
        # If perturbed is better than current, accept.
        # If perturbed is better than best, update best.
        if perturbed_ms < current_ms:
            current_perm = perturbed_perm
            current_ms = perturbed_ms
        else:
            # With some probability, accept worse solutions to escape?
            # Or just reject. ILS typically rejects worse solutions to maintain quality,
            # relying on the perturbation to find new basins.
            # However, if we always reject, we might oscillate.
            # Let's add a small probability of accepting worse solutions if we are stuck.
            # Or just keep it simple: Reject.
            pass
            
        if perturbed_ms < best_ms:
            best_perm = list(perturbed_perm)
            best_ms = perturbed_ms
            
        iteration += 1
        
        # Safety check: if we are running out of time, break
        if time.time() > global_deadline - 10.0:
             break

    return best_perm
# EVOLVE-BLOCK-END
