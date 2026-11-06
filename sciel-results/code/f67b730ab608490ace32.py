"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(order, processing_times):
    """
    Calculate the makespan for a given job order in a flow shop.
    """
    if not order:
        return 0.0
        
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
        
    # Use lists for speed in the inner loop
    curr_completion = [0.0] * n_machines
    
    for j in order:
        curr_completion[0] += processing_times[j, 0]
        for m in range(1, n_machines):
            prev_m_comp = curr_completion[m-1]
            curr_m_comp = curr_completion[m]
            if prev_m_comp > curr_m_comp:
                curr_completion[m] = prev_m_comp + processing_times[j, m]
            else:
                curr_completion[m] = curr_m_comp + processing_times[j, m]
                
    return curr_completion[-1]


def _neh_heuristic(processing_times):
    """
    Generate an initial permutation using the NEH (Nawaz-Enscore-Ham) heuristic.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Calculate sum of processing times for each job
    job_sums = processing_times.sum(axis=1)
    
    # Sort jobs in descending order of their sums
    sorted_jobs = np.argsort(-job_sums)
    
    current_sequence = []
    
    for job in sorted_jobs:
        # Find the best position to insert the current job
        best_pos = 0
        best_ms = float('inf')
        
        # Try inserting at every possible position (0 to len(current_sequence))
        for pos in range(len(current_sequence) + 1):
            new_seq = current_sequence[:pos] + [job] + current_sequence[pos:]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        # Insert the job at the best position
        current_sequence.insert(best_pos, job)
        
    return current_sequence


def _local_search_insertion(order, processing_times, deadline):
    """
    Perform insertion-based local search.
    This is often more effective than 2-opt for flow shop.
    """
    n = len(order)
    if n == 0:
        return order, 0.0
    
    current_order = order[:]
    current_ms = _makespan(current_order, processing_times)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # To avoid getting stuck in a specific cycle, we can randomize the start or just do a full pass.
        # A full pass is deterministic and safe.
        for i in range(n):
            if time.time() > deadline:
                break
            
            job = current_order[i]
            # Remove job from position i
            current_order.pop(i)
            
            best_pos = i
            best_ms = float('inf')
            
            # Try inserting at every position from i to n-1 (or 0 to n-1 for full flexibility, but i..n is faster)
            # Actually, inserting before i is equivalent to some other move, but let's check all positions 
            # for robustness, or just i..n-1 since we removed it from i.
            # Standard insertion heuristic checks all positions.
            
            for pos in range(n):
                # Insert job at pos
                current_order.insert(pos, job)
                ms = _makespan(current_order, processing_times)
                
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
                
                # Remove for next iteration
                current_order.pop(pos)
            
            # Re-insert at best position
            current_order.insert(best_pos, job)
            
            if best_ms < current_ms:
                current_ms = best_ms
                improved = True
                
            if time.time() > deadline:
                break
                
    return current_order, current_ms


def _simulated_annealing(order, processing_times, deadline, initial_temp=1000.0, cooling=0.999):
    """
    Perform Simulated Annealing to escape local minima.
    Uses adjacent swaps primarily as they are cheaper to evaluate and often sufficient.
    """
    n = len(order)
    if n == 0:
        return order, 0.0
    
    current_order = order[:]
    current_ms = _makespan(current_order, processing_times)
    best_order = current_order[:]
    best_ms = current_ms
    
    temp = initial_temp
    
    # Pre-allocate lists for speed? No, list operations are fast enough for n<1000 usually.
    
    while time.time() < deadline and temp > 1e-6:
        # Generate a random neighbor
        # 80% adjacent swap, 20% random swap (to allow larger jumps)
        if random.random() < 0.8:
            i = random.randint(0, n - 2)
            j = i + 1
        else:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
        # Perform swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Calculate new makespan
        # For adjacent swaps, we can potentially optimize, but _makespan is O(n*m).
        # Given n is likely small in synthetic instances, full recalc is fine and robust.
        new_ms = _makespan(current_order, processing_times)
        delta = new_ms - current_ms
        
        if delta < 0:
            # Accept better solution
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = current_order[:]
        else:
            # Accept worse solution with probability exp(-delta / temp)
            # Use math.exp for speed if possible, but np.exp is fine
            if random.random() < np.exp(-delta / temp):
                current_ms = new_ms
            else:
                # Reject: swap back
                current_order[i], current_order[j] = current_order[j], current_order[i]
                
        temp *= cooling
        
    return best_order, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: Generate initial solution using NEH
    current_order = _neh_heuristic(processing_times)
    current_ms = _makespan(current_order, processing_times)
    
    best_order = current_order[:]
    best_ms = current_ms
    
    # Step 2: Run local search (insertion)
    # We split the remaining time between local search and SA
    # Local search is deterministic and fast to converge, SA is stochastic
    # Let's give local search 20% of the time, SA the rest
    ls_deadline = start_time + (deadline - start_time) * 0.2
    
    ls_order, ls_ms = _local_search_insertion(current_order, processing_times, ls_deadline)
    
    if ls_ms < best_ms:
        best_ms = ls_ms
        best_order = ls_order[:]
    
    # Step 3: Run Simulated Annealing with the best solution so far
    # SA can escape local minima found by local search
    sa_deadline = deadline - 1.0  # Leave 1s margin
    
    sa_order, sa_ms = _simulated_annealing(best_order, processing_times, sa_deadline)
    
    if sa_ms < best_ms:
        best_ms = sa_ms
        best_order = sa_order[:]
    
    # Final check
    return best_order

# EVOLVE-BLOCK-END
