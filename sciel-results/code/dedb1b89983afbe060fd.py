"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, processing_times):
    """
    Calculate the makespan for a given job order in a flow shop.
    order: list of job indices
    processing_times: (n_jobs, n_machines) array
    Returns the makespan (total completion time on the last machine).
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0
    
    # C[m] is the completion time of the current job on machine m
    # We simulate the flow shop.
    # C[j][0] = C[j-1][0] + P[j][0]
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + P[j][m]
    
    # prev_completion[m] is the completion time of the previous job on machine m
    prev_completion = np.zeros(n_machines)
    
    for j in order:
        # Machine 0
        curr_completion_0 = prev_completion[0] + processing_times[j, 0]
        
        if n_machines == 1:
            prev_completion[0] = curr_completion_0
            continue
            
        # For other machines, we need the completion time on the previous machine for the current job
        # and the completion time on this machine for the previous job.
        
        # We can compute this in a loop or vectorize it?
        # Vectorizing is tricky because of the sequential dependency on the current job's previous machine completion.
        # However, for a single job, we can compute its completion times on all machines.
        
        # Let's compute the completion times for job j on all machines.
        # c_m = max(c_{m-1}, prev_completion[m]) + p_{j,m}
        
        # c_0 = prev_completion[0] + p_{j,0}
        
        # We can't easily vectorize the max with the previous element in the same array.
        # But n_machines is usually small. A loop is fine.
        
        curr_completion = np.empty(n_machines)
        curr_completion[0] = curr_completion_0
        
        for m in range(1, n_machines):
            curr_completion[m] = max(curr_completion[m-1], prev_completion[m]) + processing_times[j, m]
            
        prev_completion = curr_completion
        
    return prev_completion[-1]


def _v_shape_heuristic(processing_times):
    """
    Generate an initial permutation using the V-shaped heuristic.
    """
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    # Sum of processing times on first and last machine
    # V-shaped heuristic: jobs with high sum(p_i1, p_iM) go to the ends.
    # Specifically, sort by sum(p_i1, p_iM) in descending order.
    # Assign the largest to the first position, the next largest to the last, the next to the second, etc.
    
    sums = processing_times[:, 0] + processing_times[:, -1]
    
    # Sort job indices by sums in descending order
    sorted_jobs = np.argsort(-sums)
    
    order = [None] * n_jobs
    left = 0
    right = n_jobs - 1
    
    for i in range(n_jobs):
        job = sorted_jobs[i]
        if i % 2 == 0:
            order[left] = job
            left += 1
        else:
            order[right] = job
            right -= 1
            
    return order


def _neighborhood(order, n_jobs):
    """
    Generate neighbors by swapping two jobs.
    """
    # We will generate neighbors on the fly in the main loop to save memory and time.
    # This function is a placeholder for conceptual clarity, but we'll implement the logic inline.
    pass


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
        
    # 1. Generate initial solution using V-shaped heuristic
    current_order = _v_shape_heuristic(processing_times)
    current_makespan = _makespan(current_order, processing_times)
    
    best_order = current_order[:]
    best_makespan = current_makespan
    
    # 2. Tabu Search
    # Dynamic tabu list: store (job_i, job_j) pairs that were swapped.
    # Or store the last k moves.
    # Let's store the last `tabu_tenure` moves as a set of tuples (min(i,j), max(i,j)) or similar.
    # A better tabu representation for swaps: store the positions or the jobs involved.
    # Let's store the jobs involved in the swap. If a swap (i, j) is tabu, we cannot swap i and j again.
    # But what if i and j are swapped back? That's the point of tabu.
    # We need to distinguish between swapping job A with job B at position x, and swapping them elsewhere?
    # In permutation flow shop, swapping two jobs changes the sequence.
    # If we swap job at index i and job at index j, the tabu attribute is usually the pair of jobs (or their positions).
    # Let's use the pair of job indices.
    
    tabu_list = set()
    tabu_tenure = 20  # Dynamic? Let's start with a fixed one and maybe adjust.
    # Dynamic tabu list: if no improvement for some iterations, increase tenure? Or decrease?
    # Let's keep it simple first: fixed tenure, but maybe adjust based on stagnation.
    
    no_improve_count = 0
    max_no_improve = 50
    current_tenure = 20
    
    while time.time() < deadline:
        # Generate a random neighbor by swapping two jobs
        # To diversify, we can pick random indices i, j
        i = np.random.randint(n_jobs)
        j = np.random.randint(n_jobs)
        if i == j:
            continue
            
        # Check if this swap is tabu
        swap_key = (min(current_order[i], current_order[j]), max(current_order[i], current_order[j]))
        # Wait, this key is based on job indices, not positions.
        # If we swap job A and job B, the key is (A, B).
        # This prevents swapping A and B again, regardless of position.
        # Is this good? It might be too restrictive if A and B are far apart and swapping them is beneficial in a different context?
        # In a permutation, the relative order of A and B is what matters for their contribution to the flow shop?
        # Actually, the makespan depends on the entire sequence.
        # Swapping A and B changes the sequence.
        # If we ban swapping A and B, we ban any move that results in A and B being swapped relative to each other?
        # No, swapping A and B at positions i and j is a specific move.
        # If we use job indices as the tabu key, we are saying "Do not swap job A and job B".
        # This is a valid tabu definition.
        
        if swap_key in tabu_list:
            # Check aspiration criterion: if the move improves the best solution, allow it
            # For now, let's just skip tabu moves unless they improve the best.
            # To check if it improves the best, we'd need to evaluate it, which is expensive.
            # Let's just skip for now.
            continue
        
        # Perform the swap
        new_order = current_order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        # Evaluate
        new_makespan = _makespan(new_order, processing_times)
        
        # Update tabu list
        tabu_list.add(swap_key)
        if len(tabu_list) > current_tenure:
            # Remove the oldest element?
            # Sets don't have order. We need a queue or a list to track order.
            # Let's use a list for the tabu list to maintain order, and a set for fast lookup.
            # This is getting complex. Let's simplify:
            # Use a list `tabu_moves` of (i, j) positions? Or job pairs?
            # If we use positions, swapping back is immediately tabu?
            # Let's stick to job pairs and use a list for FIFO removal.
            # But we already added to a set.
            # Let's refactor to use a list and a set.
            
            # For now, let's just clear the list if it gets too big? That's not dynamic.
            # Let's change the structure.
            
            pass # Placeholder for proper tabu management
            
        # Actually, let's re-implement the tabu list properly.
        # We'll break out of this loop iteration and restart the tabu logic in the next iteration?
        # No, let's just use a simple list-based tabu list.
        
        # Let's restart the loop structure to handle the tabu list correctly.
        break # Exit this flawed loop and rewrite the search part
        
    # Rewrite the search part properly
    
    # Re-initialize state
    current_order = best_order[:] # Start from the best found so far (which is the initial)
    current_makespan = best_makespan
    
    tabu_moves = [] # List of (job1, job2) tuples
    tabu_set = set() # For O(1) lookup
    current_tenure = 20
    
    no_improve_count = 0
    
    while time.time() < deadline:
        # Generate random move
        i = np.random.randint(n_jobs)
        j = np.random.randint(n_jobs)
        if i == j:
            continue
            
        job_i = current_order[i]
        job_j = current_order[j]
        swap_key = (min(job_i, job_j), max(job_i, job_j))
        
        is_tabu = swap_key in tabu_set
        
        # Perform swap to evaluate
        new_order = current_order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        new_makespan = _makespan(new_order, processing_times)
        
        # Aspiration criterion: allow tabu move if it improves the best solution
        if is_tabu and new_makespan >= best_makespan:
            continue
            
        # Accept the move
        current_order = new_order
        current_makespan = new_makespan
        
        # Update best
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_order = current_order[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Update tabu list
        tabu_moves.append(swap_key)
        tabu_set.add(swap_key)
        
        # Dynamic tenure: if stagnating, increase tenure to diversify?
        # Or decrease to intensify?
        # Usually, if no improvement, we might want to explore more (increase tenure)
        # or accept worse moves more often.
        # Let's try increasing tenure slightly.
        if no_improve_count > 0 and no_improve_count % 10 == 0:
            current_tenure = min(current_tenure + 5, 100)
            
        # Remove old moves
        while len(tabu_moves) > current_tenure:
            old_move = tabu_moves.pop(0)
            tabu_set.remove(old_move)
            
    return best_order

# EVOLVE-BLOCK-END
