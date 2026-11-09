"""Improved program for flow-shop permutation construction using NEH + Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = P.shape[1]
    
    # C[m] is the completion time of the current job on machine m
    # We iterate through jobs in the order, updating completion times.
    # Let C[m] be the completion time of the previous job on machine m.
    # For the current job j, completion on machine m is max(C[m], completion_on_prev_machine_for_j) + P[j, m]
    
    # Vectorized-ish approach for speed if possible, but loop is fine for n~100-500
    # Let's use a loop over jobs and inner loop over machines for clarity and correctness.
    # Actually, we can optimize by keeping track of the completion time of the current job on the previous machine.
    
    c_prev_machine = 0.0
    # c_machines[m] holds the completion time of the *last* job processed on machine m
    c_machines = np.zeros(n_machines)
    
    for j in order:
        # Process job j on all machines
        # completion on machine 0: max(c_machines[0], 0) + P[j, 0]
        # completion on machine m: max(c_machines[m], completion_on_m-1) + P[j, m]
        
        # We can do this in a loop over machines
        # Or, since we need the final makespan, we just need the last machine's completion time for the last job.
        # But we must update c_machines for all machines.
        
        # Optimization: 
        # c_machines[0] = c_machines[0] + P[j, 0] (since c_prev_machine starts at 0, max(c_machines[0], 0) is just c_machines[0] if we assume non-negative times, but wait, c_machines[0] is completion of previous job. So max is correct. But if it's the first job, c_machines[0] is 0. So c_machines[0] + P[j,0] is correct? No, max(c_machines[0], 0) + P[j,0]. Since times are positive, c_machines[0] >= 0. So yes.
        # But for m>0, it's max(c_machines[m], c_prev_machine) + P[j, m].
        
        # Let's just loop. It's O(N*M). For N=500, M=10, that's 5000 ops per makespan eval. 
        # If we do 10000 evals, that's 50M ops. NumPy is faster, but Python loops might be okay within 150s if we are careful.
        # Let's try to vectorize the inner machine loop? Hard because of the dependency.
        # Let's stick to a clean Python loop first, then optimize if needed.
        
        # Actually, we can use a trick.
        # c_machines[m] = max(c_machines[m], c_machines[m-1]) + P[j, m] ? No, c_machines[m-1] is the completion of the PREVIOUS job on machine m-1.
        # The dependency is on the CURRENT job's completion on m-1.
        
        # Let's use a simple loop.
        # c_curr = 0
        # for m in range(n_machines):
        #     c_curr = max(c_machines[m], c_curr) + P[j, m]
        #     c_machines[m] = c_curr
        # makespan = c_curr (of the last job)
        
        c_curr = 0.0
        # Accessing P[j, m] in a loop is slow. Let's get the row first.
        row = P[j]
        for m in range(n_machines):
            c_curr = max(c_machines[m], c_curr) + row[m]
            c_machines[m] = c_curr
            
    return c_curr

def _neh_heuristic(P):
    """
    NEH (Nawaz-Enscore-Ham) heuristic.
    1. Sort jobs by total processing time in descending order.
    2. Insert each job into the best position in the current sequence to minimize makespan.
    """
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    # Sort indices by total time descending
    order_indices = np.argsort(-totals)
    
    current_seq = []
    for idx in order_indices:
        # Try all possible insertion positions
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_seq) + 1):
            # Create new sequence
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            # Calculate makespan
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
        
    return current_seq

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using NEH heuristic followed by local search.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # 1. Initialize with NEH
    best_order = _neh_heuristic(P)
    best_makespan = _makespan(best_order, P)
    
    # 2. Local Search: 2-Opt / Insertion
    # We will try swapping adjacent jobs or moving a job to a different position.
    # To be efficient, we can limit the search or use a fast evaluation.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try all possible 2-opt moves (swap i, i+1) or insertion moves?
        # Insertion is more powerful. Let's do insertion.
        # For each job i, try moving it to every other position j.
        
        # Optimization: If we move job i to j, we can calculate the new makespan more efficiently?
        # Not easily without complex data structures. Let's just recompute.
        # But recomputing O(N*M) for O(N^2) moves is O(N^3*M).
        # For N=100, 100^3 * 10 = 10^7 ops. That's fast.
        # For N=500, 500^3 * 10 = 1.25 * 10^9 ops. That might be too slow in Python.
        
        # Let's check N.
        if n_jobs > 200:
            # Use a simpler neighborhood or limit iterations
            # Maybe just check adjacent swaps? O(N) moves, each O(N*M). Total O(N^2*M).
            # Or limit the number of passes.
            pass_count = 0
            max_passes = 50
            while improved and pass_count < max_passes and (time.time() - start_time) < time_limit:
                improved = False
                pass_count += 1
                for i in range(n_jobs - 1):
                    # Swap i and i+1
                    new_order = best_order.copy()
                    new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
                    ms = _makespan(new_order, P)
                    if ms < best_makespan:
                        best_makespan = ms
                        best_order = new_order
                        improved = True
        else:
            # Full insertion neighborhood
            for i in range(n_jobs):
                job_i = best_order[i]
                # Remove job_i
                remaining = best_order[:i] + best_order[i+1:]
                for j in range(len(remaining) + 1):
                    if j == i or j == i + 1:
                        continue # Skip same position and adjacent (already covered by swap? No, insertion covers all)
                    # Actually, inserting at j in remaining is equivalent to inserting at j in original if j < i, or j+1 if j >= i.
                    # Let's just construct the new sequence.
                    new_order = remaining[:j] + [job_i] + remaining[j:]
                    ms = _makespan(new_order, P)
                    if ms < best_makespan:
                        best_makespan = ms
                        best_order = new_order
                        improved = True
                        break # Restart search from beginning? Or continue?
                        # Usually, first-improvement or best-improvement.
                        # Let's do first-improvement for speed, but restart the outer loop?
                        # Or just break inner and continue outer?
                        # If we break inner, we should probably restart the whole pass to ensure we find all improvements.
                        # But for time budget, let's just break inner and continue.
                    
    return best_order
# EVOLVE-BLOCK-END
