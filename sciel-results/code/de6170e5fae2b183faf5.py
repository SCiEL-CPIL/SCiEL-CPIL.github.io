"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    # Helper to calculate makespan for a given permutation
    def calc_makespan(perm):
        # perm is a list of job indices
        # completion[m] is the time machine m is free
        # We simulate the flow shop
        # Note: Using a list for completion is faster than array for small M in Python loops?
        # Let's use a list for speed in inner loop.
        completion = [0.0] * n_machines
        for j in perm:
            job_times = processing_times[j]
            for m in range(n_machines):
                # Flow shop constraint:
                # Machine m can start job j only after:
                # 1. Machine m finished previous job (completion[m])
                # 2. Machine m-1 finished this job j (completion[m-1])
                if m == 0:
                    start_time_j = completion[0]
                else:
                    start_time_j = max(completion[m], completion[m-1])
                completion[m] = start_time_j + job_times[m]
        return completion[n_machines - 1]

    # 1. Initial Solution: NEH Heuristic
    # Sort jobs by decreasing total processing time
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    current_perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        # Insert job into all possible positions
        for i in range(len(current_perm) + 1):
            # Create trial permutation
            trial = current_perm[:i] + [job] + current_perm[i:]
            ms = calc_makespan(trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = i
        current_perm.insert(best_pos, job)
        
    best_perm = current_perm[:]
    best_ms = calc_makespan(best_perm)
    
    # 2. Tabu Search with Insertion Moves
    # Tabu list stores (job_removed, pos_inserted) or just the move signature
    # To avoid cycling, we tabu the move (job, old_pos, new_pos) or simply
    # tabu the job for T iterations (simple strategy).
    # Better: Tabu the reverse move.
    # Let's use a simple approach: Tabu list of (job, position) pairs recently moved to.
    # Or simply tabu the job index for T iterations. This is less precise but fast.
    # Let's use a list of the last T moves: (job, from_pos, to_pos)
    
    tabu_list = []
    tabu_tenure = 10  # Adjust based on J? Fixed 10 is safe for small J.
    
    iter_count = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        iter_count += 1
        
        # Generate neighbors: Insertion moves
        # For each job, remove it and insert at every other position
        # This is O(J^2) neighbors.
        
        best_neighbor_perm = None
        best_neighbor_ms = float('inf')
        best_move = None
        
        current_len = len(current_perm)
        
        for i in range(current_len):
            job = current_perm[i]
            # Remove job
            remaining = current_perm[:i] + current_perm[i+1:]
            
            for k in range(current_len - 1):
                # Skip if k is the original position (or adjacent effectively same)
                # Original position i.
                # If k == i, it's the same.
                # If k == i-1, it's the same (inserting before i is same as removing i and inserting at i-1? No.)
                # Let's just check all k.
                
                # Tabu check: Is (job, k) in tabu list?
                # We store (job, target_pos) in tabu list.
                # Actually, tabu should prevent reversing the last move.
                # Last move was (job, from, to). Reverse is (job, to, from).
                # So if we move job 'job' to position 'k', we should check if (job, k) is tabu.
                
                is_tabu = False
                for t_move in tabu_list:
                    if t_move[0] == job and t_move[1] == k:
                        is_tabu = True
                        break
                
                if is_tabu:
                    # Aspiration: if this move improves global best, allow it
                    # We don't know ms yet, so we skip for now to save time,
                    # or we evaluate and check.
                    # For speed, we skip tabu moves unless we are desperate.
                    # Let's just skip.
                    continue

                # Construct new permutation
                # Insert job at k in remaining
                new_perm = remaining[:k] + [job] + remaining[k:]
                
                # Evaluate
                ms = calc_makespan(new_perm)
                
                if ms < best_neighbor_ms:
                    best_neighbor_ms = ms
                    best_neighbor_perm = new_perm
                    best_move = (job, k)
                    
        # If no non-tabu move found, pick best move regardless of tabu
        if best_move is None:
            for i in range(current_len):
                job = current_perm[i]
                remaining = current_perm[:i] + current_perm[i+1:]
                for k in range(current_len - 1):
                    new_perm = remaining[:k] + [job] + remaining[k:]
                    ms = calc_makespan(new_perm)
                    if ms < best_neighbor_ms:
                        best_neighbor_ms = ms
                        best_neighbor_perm = new_perm
                        best_move = (job, k)
            
            if best_move is None:
                break # No moves possible?
        
        # Update Tabu List
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        
        # Update Current and Best
        current_perm = best_neighbor_perm
        
        if best_neighbor_ms < best_ms:
            best_ms = best_neighbor_ms
            best_perm = current_perm[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Simple diversification: if stuck for long, restart or random move?
            # For now, just continue.
            
        # Safety: if J is very large, this might be slow.
        # If J > 100, this loop is too slow.
        # We assume J is small enough.
        # If we are getting close to deadline, break.
        
    return best_perm

# EVOLVE-BLOCK-END
