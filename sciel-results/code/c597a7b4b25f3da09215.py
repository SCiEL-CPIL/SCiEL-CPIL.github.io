"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

# Use a fixed seed for reproducibility within a single call, but we can vary it
# if we want, though for a single call it doesn't matter much unless we do restarts.
# Actually, since the function is called once per instance, we just need a good algorithm.

def _makespan(perm, P):
    """Compute makespan for a given permutation."""
    # P is (n_jobs, n_machines)
    # perm is list of job indices
    n_machines = P.shape[1]
    completion = np.zeros(n_machines)
    for job in perm:
        for m in range(n_machines):
            completion[m] += P[job, m]
    return completion[-1]

def _neh_order(P):
    """
    NEH heuristic:
    1. Sort jobs by total processing time (descending).
    2. Insert each job into the position that minimizes the makespan.
    """
    n_jobs = P.shape[0]
    n_machines = P.shape[1]
    
    # Step 1: Sort by total processing time
    totals = P.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    # Step 2: Insertive construction
    perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_mk = float('inf')
        
        # Try inserting at each position
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            mk = _makespan(new_perm, P)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        
        perm.insert(best_pos, job)
    
    return perm

def _improve_permutation(perm, P, max_iters=5000, time_limit=5.0):
    """
    Local search using insertion moves.
    """
    n_jobs = len(perm)
    n_machines = P.shape[1]
    current_perm = list(perm)
    current_mk = _makespan(current_perm, P)
    
    start_time = time.time()
    no_improve_count = 0
    
    # Pre-compute something? No, just do simple insertion.
    # To speed up, we can calculate delta makespan efficiently, but for small n, full recalc is fine.
    # Let's optimize the makespan calculation for speed.
    
    def calc_mk(p):
        comp = np.zeros(n_machines)
        for j in p:
            # comp[m] = max(comp[m], comp[m-1]) + P[j, m]  -- wait, this is for flow shop.
            # Actually:
            # C[j][m] = max(C[j][m-1], C[j-1][m]) + P[j][m]
            # We only need the last row (last job's completion times) to get the makespan?
            # No, we need the completion time of the last job on the last machine.
            # But to compute C[j][m], we need C[j-1][m] and C[j][m-1].
            # So we can just keep an array `prev_row` for the previous job's completion times.
            pass
        
        # Efficient makespan calculation:
        prev_row = np.zeros(n_machines)
        for j in p:
            curr_row = np.empty(n_machines)
            curr_row[0] = prev_row[0] + P[j, 0]
            for m in range(1, n_machines):
                curr_row[m] = max(prev_row[m], curr_row[m-1]) + P[j, m]
            prev_row = curr_row
        return prev_row[-1]
    
    current_mk = calc_mk(current_perm)
    
    while no_improve_count < 500:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Randomly pick a job and insert it elsewhere
        i = random.randint(0, n_jobs - 1)
        job = current_perm.pop(i)
        
        # Find best position to re-insert
        best_pos = i
        best_mk = current_mk
        
        for pos in range(n_jobs):
            if pos == i:
                continue
            # Insert at pos
            new_perm = current_perm[:pos] + [job] + current_perm[pos:]
            mk = calc_mk(new_perm)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        
        if best_mk < current_mk:
            current_perm.insert(best_pos, job)
            current_mk = best_mk
            no_improve_count = 0
            improved = True
        else:
            # Revert
            current_perm.insert(i, job)
            no_improve_count += 1
            
    return current_perm, current_mk

def order_jobs(processing_times):
    P = np.asarray(processing_times)
    n_jobs = P.shape[0]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Deadline guard
    total_time_limit = 110.0  # Leave 10s margin for 120s limit
    start_global = time.time()
    
    # 1. Generate initial solution using NEH
    best_perm = _neh_order(P)
    best_mk = _makespan(best_perm, P)
    
    # 2. Improve with local search
    best_perm, best_mk = _improve_permutation(best_perm, P, time_limit=10.0)
    
    # 3. If time remains, do more iterations or restarts?
    # The current strategy is simple. Let's see if we can do multiple NEH restarts with different randomizations?
    # NEH is deterministic. We can randomize the sorting slightly or the insertion order?
    # Or just run local search for longer if time permits.
    
    elapsed = time.time() - start_global
    remaining_time = total_time_limit - elapsed
    
    if remaining_time > 5.0:
        # Run more local search iterations
        # We can just call _improve_permutation again with a tighter time limit
        best_perm, best_mk = _improve_permutation(best_perm, P, time_limit=remaining_time - 2.0)
    
    return list(best_perm)

# EVOLVE-BLOCK-END
