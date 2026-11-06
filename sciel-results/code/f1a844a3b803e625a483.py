"""
Strategy: Systematic Best-Insertion Construction + Exhaustive Neighborhood Search
This approach constructs the permutation by greedily inserting each job into the
position that minimizes the makespan. It then refines the solution using a
systematic search over all possible single-job moves and swaps, rather than
relying on random sampling. This exploits the structure of the flow-shop problem
more directly than random restarts.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(perm, processing_times):
    """
    Compute makespan for a given permutation.
    perm: list of job indices
    processing_times: numpy array of shape (n_jobs, n_machines)
    Returns: float makespan
    """
    n_machines = processing_times.shape[1]
    if not perm:
        return 0.0
    
    # C[j][m] is completion time of job j on machine m
    # We only need the previous row's completion times to compute the next.
    # C_prev[m] = completion time of previous job on machine m
    # C_curr[m] = completion time of current job on machine m
    
    C_prev = np.zeros(n_machines)
    makespan = 0.0
    
    for job_idx in perm:
        C_curr = np.zeros(n_machines)
        for m in range(n_machines):
            if m == 0:
                C_curr[m] = C_prev[m] + processing_times[job_idx, m]
            else:
                C_curr[m] = max(C_prev[m], C_curr[m-1]) + processing_times[job_idx, m]
        C_prev = C_curr
        makespan = C_prev[-1]
        
    return makespan

def _makespan_from_array(perm_arr, processing_times):
    """
    Optimized makespan calculation for numpy array input.
    """
    n_machines = processing_times.shape[1]
    if len(perm_arr) == 0:
        return 0.0
        
    C_prev = np.zeros(n_machines)
    makespan = 0.0
    
    for job_idx in perm_arr:
        # Vectorized update where possible? 
        # C_curr[0] = C_prev[0] + p[job, 0]
        # C_curr[m] = max(C_prev[m], C_curr[m-1]) + p[job, m]
        # This dependency chain makes full vectorization hard without loops or scanning.
        # However, for small n_machines, the loop is fine.
        
        p_job = processing_times[job_idx]
        C_curr = np.empty(n_machines)
        C_curr[0] = C_prev[0] + p_job[0]
        if n_machines > 1:
            # C_curr[1:] = np.maximum(C_prev[1:], np.roll(C_curr, 1)[1:]) + p_job[1:]
            # This is tricky because C_curr[m-1] is from the CURRENT row.
            # We must compute sequentially.
            for m in range(1, n_machines):
                C_curr[m] = max(C_prev[m], C_curr[m-1]) + p_job[m]
        C_prev = C_curr
        makespan = C_prev[-1]
        
    return makespan

def _construct_best_insertion(processing_times, n_jobs, rng):
    """
    Construct a permutation by inserting jobs one by one.
    At each step, try inserting the next job into all possible positions and pick the best.
    To ensure diversity, we can randomize the order in which we select the "next job" to insert,
    or use a greedy selection based on total processing time (NEH-like) but with exact placement.
    Here we use a mix: NEH order for job selection, but exact best-position insertion.
    """
    n_jobs, n_machines = processing_times.shape
    
    # Determine the order in which we add jobs to the permutation.
    # Standard NEH sorts by total processing time descending.
    totals = processing_times.sum(axis=1)
    job_order = list(np.argsort(-totals))
    
    # We can also try a few variations of job_order to get diverse initial solutions
    # but for now, let's stick to one strong constructive path.
    
    current_perm = []
    
    for job in job_order:
        best_cost = float('inf')
        best_pos = 0
        
        # Try inserting 'job' at each position in current_perm
        for pos in range(len(current_perm) + 1):
            # Create a new permutation list
            # Optimization: Instead of creating lists, we can calculate makespan incrementally?
            # No, makespan is global. But N is likely small (< 50-100).
            # Creating a list of size k is O(k). Doing it k times is O(k^2).
            # Total construction is O(N^3) roughly. For N=50, 125k ops * N_machines. Very fast.
            
            new_perm = current_perm[:pos] + [job] + current_perm[pos:]
            cost = _makespan(new_perm, processing_times)
            
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        
        # Insert at best position
        current_perm.insert(best_pos, job)
        
    return current_perm

def _systematic_search(perm, processing_times, deadline):
    """
    Perform systematic neighborhood search.
    1. Try all single-job moves (Or-opt of size 1).
    2. Try all swaps of two jobs.
    Repeat until no improvement or deadline.
    """
    current_perm = list(perm)
    current_cost = _makespan(current_perm, processing_times)
    n_jobs = len(current_perm)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try all single-job moves
        best_move = None
        best_cost = current_cost
        
        for i in range(n_jobs):
            job = current_perm[i]
            # Remove job
            rest = current_perm[:i] + current_perm[i+1:]
            
            for j in range(len(rest) + 1):
                # Skip if it's the same position (j == i or j == i+1 effectively)
                # If we removed at i, inserting at i puts it back. Inserting at i+1 also puts it back.
                # We want to try all distinct positions.
                if j == i or j == i + 1:
                    continue
                
                new_perm = rest[:j] + [job] + rest[j:]
                cost = _makespan(new_perm, processing_times)
                
                if cost < best_cost:
                    best_cost = cost
                    best_move = (i, j, job, 'move')
                    
        if best_move:
            i, j, job, _ = best_move
            # Apply move
            rest = current_perm[:i] + current_perm[i+1:]
            current_perm = rest[:j] + [job] + rest[j:]
            current_cost = best_cost
            improved = True
            continue # Restart search with new incumbent
        
        # 2. Try all swaps
        best_swap = None
        best_swap_cost = current_cost
        
        for i in range(n_jobs):
            for j in range(i + 1, n_jobs):
                new_perm = current_perm[:]
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                cost = _makespan(new_perm, processing_times)
                
                if cost < best_swap_cost:
                    best_swap_cost = cost
                    best_swap = (i, j)
                    
        if best_swap:
            i, j = best_swap
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            current_cost = best_swap_cost
            improved = True
            continue
            
    return current_perm, current_cost

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on local synthetic flow-shop instances.
    """
    start_time = time.time()
    # Safety margin: return 2 seconds before the 150s limit
    deadline = start_time + 148.0
    
    n_jobs, n_machines = processing_times.shape
    rng = random.Random(42)
    
    best_perm = None
    best_cost = float('inf')
    
    # Strategy:
    # 1. Generate multiple high-quality initial solutions using Best-Insertion with different job selection orders.
    # 2. Run systematic local search on the best one(s).
    
    # Initial Solution 1: NEH order (Total PT descending)
    # Initial Solution 2: SPT order (Total PT ascending)
    # Initial Solution 3: Random order (but then best-insertion construction? No, best-insertion constructs from scratch).
    # Actually, _construct_best_insertion takes the job_order.
    # Let's generate a few different job_orders to feed into the constructor.
    
    job_orders = []
    
    # 1. NEH
    totals = processing_times.sum(axis=1)
    job_orders.append(list(np.argsort(-totals)))
    
    # 2. SPT
    job_orders.append(list(np.argsort(totals)))
    
    # 3. Random
    rand_order = list(range(n_jobs))
    rng.shuffle(rand_order)
    job_orders.append(rand_order)
    
    # 4. Maybe a second random
    rand_order2 = list(range(n_jobs))
    rng.shuffle(rand_order2)
    job_orders.append(rand_order2)
    
    # Construct solutions
    candidates = []
    for order in job_orders:
        if time.time() > deadline - 5:
            break
        perm = _construct_best_insertion(processing_times, n_jobs, rng)
        cost = _makespan(perm, processing_times)
        candidates.append((cost, perm))
        
    # Sort candidates by cost
    candidates.sort(key=lambda x: x[0])
    
    # Run systematic search on the best candidate
    if candidates:
        best_cost, best_perm = candidates[0]
        # We can also run search on the second best if time permits
        # But systematic search is exhaustive, so it's likely to converge to a good local opt.
        
        # Let's run search on the top 1 or 2 candidates
        for i in range(min(2, len(candidates))):
            if time.time() > deadline - 2:
                break
            
            cand_cost, cand_perm = candidates[i]
            # Only search if it's close to the best or if it's the first one
            if i == 0 or cand_cost < best_cost * 1.05:
                refined_perm, refined_cost = _systematic_search(cand_perm, processing_times, deadline)
                if refined_cost < best_cost:
                    best_cost = refined_cost
                    best_perm = refined_perm
                    
    # Fallback if something went wrong (shouldn't happen)
    if best_perm is None:
        best_perm = list(range(n_jobs))
        
    return list(best_perm)
# EVOLVE-BLOCK-END
