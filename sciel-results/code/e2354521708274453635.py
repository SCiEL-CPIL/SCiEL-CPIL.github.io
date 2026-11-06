"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    
    # Efficient makespan computation
    # C[j][m] = completion time of job j on machine m
    # We can compute this incrementally
    order = np.array(order, dtype=int)
    # processing_times for the jobs in order
    pt = processing_times[order]  # shape (n_jobs, n_machines)
    
    # Use the standard flow shop makespan formula
    # C_jm = max(C_j,m-1, C_j-1,m) + P_jm
    # We only need the final row (last machine) for makespan
    # But we need all intermediate values
    
    # Vectorized approach for small n_jobs
    # For larger, we can use the recurrence
    if n_jobs == 1:
        return float(np.sum(pt[0]))
    
    # Build completion time matrix
    # C has shape (n_jobs, n_machines)
    C = np.zeros((n_jobs, n_machines))
    C[0, 0] = pt[0, 0]
    
    # First row: cumulative sum along machines
    for m in range(1, n_machines):
        C[0, m] = C[0, m-1] + pt[0, m]
    
    # First column: cumulative sum along jobs
    for j in range(1, n_jobs):
        C[j, 0] = C[j-1, 0] + pt[j, 0]
    
    # Fill in the rest
    for j in range(1, n_jobs):
        for m in range(1, n_machines):
            C[j, m] = max(C[j-1, m], C[j, m-1]) + pt[j, m]
    
    return float(C[-1, -1])


def _makespan_delta_swap(processing_times, order, i, j):
    """
    Compute the delta in makespan when swapping positions i and j in order.
    This is an approximation that only considers the local neighborhood.
    For exact computation, we'd need to recompute, but for large n this is expensive.
    
    Instead, let's just compute the full makespan for the swapped order.
    """
    new_order = list(order)
    new_order[i], new_order[j] = new_order[j], new_order[i]
    return _makespan(processing_times, new_order)


def _neh_initialization(processing_times):
    """
    NEH (Nawaz, Enscore, Ham) heuristic for initial solution.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    order = []
    for job in sorted_jobs:
        # Try inserting job at each position and pick the best
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(order) + 1):
            candidate = order[:pos] + [job] + order[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        order.insert(best_pos, int(job))
    
    return order


def _insertion_local_search(processing_times, order, max_iter=1000, deadline=None):
    """
    Insertion-based local search: move each job to the best position.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return order
    
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    improved = True
    
    while improved:
        improved = False
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
            job = current_order[i]
            # Try inserting job at each other position
            best_pos = i
            best_ms = current_ms
            
            for j in range(n_jobs):
                if j == i:
                    continue
                # Remove job from position i and insert at position j
                # Adjust index if j > i
                if j > i:
                    new_order = current_order[:i] + current_order[i+1:j] + [job] + current_order[j:]
                else:
                    new_order = current_order[:j] + [job] + current_order[j:i] + current_order[i+1:]
                
                ms = _makespan(processing_times, new_order)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = j
            
            if best_pos != i:
                # Move job to best position
                job = current_order[i]
                if best_pos > i:
                    current_order = current_order[:i] + current_order[i+1:best_pos] + [job] + current_order[best_pos:]
                else:
                    current_order = current_order[:best_pos] + [job] + current_order[best_pos:i] + current_order[i+1:]
                current_ms = best_ms
                improved = True
        
        if deadline is not None and time.time() > deadline:
            break
    
    return current_order


def _perturb_order(order, strength=0.1):
    """
    Perturb an order by swapping a random number of pairs.
    """
    n = len(order)
    if n <= 1:
        return list(order)
    
    perturbed = list(order)
    n_swaps = max(1, int(strength * n))
    
    for _ in range(n_swaps):
        i, j = random.sample(range(n), 2)
        perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
    
    return perturbed


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses NEH initialization, multi-start with perturbations, and insertion local search.
    """
    processing_times = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # NEH initialization
    best_order = _neh_initialization(processing_times)
    best_ms = _makespan(processing_times, best_order)
    
    # Local search on the initial solution
    best_order = _insertion_local_search(processing_times, best_order, deadline=deadline)
    best_ms = _makespan(processing_times, best_order)
    
    if time.time() > deadline:
        return best_order
    
    # Multi-start with diverse perturbations
    random.seed(42)  # For reproducibility
    n_restarts = 0
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() < deadline:
        # Create a perturbed version of the best solution
        strength = 0.1 + 0.2 * (n_restarts % 5) / 5.0
        perturbed = _perturb_order(best_order, strength=strength)
        
        # Local search on the perturbed solution
        candidate_order = _insertion_local_search(processing_times, perturbed, deadline=deadline)
        candidate_ms = _makespan(processing_times, candidate_order)
        
        if candidate_ms < best_ms:
            best_ms = candidate_ms
            best_order = candidate_order
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        n_restarts += 1
        
        # Stop if we've had many consecutive non-improving restarts and time is running low
        if no_improve_count >= max_no_improve and time.time() > start_time + 30:
            # Try a few more with stronger perturbation
            if no_improve_count >= max_no_improve * 2:
                break
        
        if time.time() > deadline:
            break
    
    return best_order
# EVOLVE-BLOCK-END
