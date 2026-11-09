"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """
    Compute makespan for a given job order.
    Uses the standard recurrence relation for flow shop:
    C[i][j] = max(C[i-1][j], C[i][j-1]) + p[i][j]
    We only need to track the current row of C.
    """
    if not order:
        return 0.0
    n_machines = processing_times.shape[1]
    
    # c[j] stores C[current_job][j]
    c = np.zeros(n_machines)
    
    for job_idx in order:
        p_row = processing_times[job_idx]
        # Update c in place
        # c[0] = c[0] + p[0]
        # c[j] = max(c[j], c[j-1]) + p[j]
        # We must iterate j from 0 to M-1
        # Note: c[j-1] is already updated for the current job
        # c[j] is still the value from the previous job
        for j in range(n_machines):
            if j == 0:
                c[j] += p_row[j]
            else:
                # max(old_c[j], new_c[j-1]) + p[j]
                if c[j] > c[j-1]:
                    c[j] += p_row[j]
                else:
                    c[j] = c[j-1] + p_row[j]
    return c[-1]

def _makespan_from_c(processing_times, order, c_start):
    """
    Compute makespan starting from a given completion time vector c_start.
    c_start represents the completion times of the machines before the first job in 'order'.
    """
    if not order:
        return c_start[-1] if len(c_start) > 0 else 0.0
    n_machines = processing_times.shape[1]
    c = c_start.copy()
    
    for job_idx in order:
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            if j == 0:
                c[j] += p_row[j]
            else:
                if c[j] > c[j-1]:
                    c[j] += p_row[j]
                else:
                    c[j] = c[j-1] + p_row[j]
    return c[-1]

def _neh_initialization(processing_times):
    """
    NEH (Nawaz-Enscore-Ham) heuristic.
    1. Sort jobs in decreasing order of total processing time.
    2. Insert each job into the position that minimizes the makespan of the partial schedule.
    """
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    current_order = [sorted_jobs[0]]
    for job in sorted_jobs[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current_order) + 1):
            # Create new order
            # Optimization: We can compute makespan incrementally if we had the C matrix,
            # but for simplicity and robustness in initialization, we use full calc.
            new_order = current_order[:pos] + [job] + current_order[pos:]
            ms = _makespan(processing_times, new_order)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current_order.insert(best_pos, job)
        
    return current_order

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialization
    current_order = _neh_initialization(processing_times)
    current_ms = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_ms = current_ms
    
    # 2. Tabu Search
    # Tabu list: set of (job_idx, position) that are forbidden to move TO or FROM?
    # Standard: If we move job i from pos a to pos b, we tabu (i, b) for T iterations.
    # Also tabu (i, a) to prevent immediate undo? Or just (i, b).
    # Let's use (job_idx, target_pos).
    
    tabu_tenure = 10 + n_jobs // 10  # Adaptive tenure
    tabu_list = {}  # key: (job_idx, pos), value: iteration_when_expired
    iteration = 0
    no_improve_count = 0
    perturbation_threshold = 50  # If stuck for 50 iters, perturb
    
    while time.time() < deadline:
        iteration += 1
        
        # Check for perturbation
        if no_improve_count > perturbation_threshold:
            # Random Perturbation: Swap 2-3 random jobs
            if n_jobs > 2:
                num_swaps = random.randint(1, 2)
                for _ in range(num_swaps):
                    i = random.randint(0, n_jobs - 1)
                    j = random.randint(0, n_jobs - 1)
                    if i != j:
                        current_order[i], current_order[j] = current_order[j], current_order[i]
                current_ms = _makespan(processing_times, current_order)
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_order = list(current_order)
                no_improve_count = 0
                # Clear tabu list to allow escape
                tabu_list.clear()
                continue

        # Generate Neighborhood: Insertion Moves
        n = len(current_order)
        best_move_ms = float('inf')
        best_move_job = -1
        best_move_from = -1
        best_move_to = -1
        best_is_tabu = False
        
        # We evaluate all possible insertion moves
        # For each job i, try moving it to every position j != i
        for i in range(n):
            job_idx = current_order[i]
            # Remove job from i
            # We need to evaluate the makespan for each insertion position.
            # To avoid O(N^2 * M) full recalculations for every i,
            # we can be smarter, but for N < 100, O(N^2 * M) with tight loops is okay.
            # Let's optimize by only checking positions that are likely better?
            # No, stick to full neighborhood for robustness, but break early if time is tight.
            
            if time.time() > deadline - 5.0:
                break
                
            # Skip if job is already tabu to move from i? No, tabu is usually on the destination.
            # Let's check all destinations
            for j in range(n):
                if i == j:
                    continue
                
                # Check Tabu Status
                # Key: (job_idx, j)
                key = (job_idx, j)
                if key in tabu_list and tabu_list[key] > iteration:
                    # It is tabu
                    is_tabu = True
                else:
                    is_tabu = False
                
                # Calculate new makespan
                # Move job i to j
                # If i < j: remove i, then insert at j (index shifts)
                # If i > j: remove i, then insert at j
                
                # To save time, we can construct the new order and calc makespan.
                # Optimization: If we just swapped, we could use delta, but insertion is complex.
                # We will use a helper that creates the list.
                
                # List slicing is O(N). _makespan is O(N*M).
                # This is O(N^2 * M) per iteration.
                # For N=50, M=10, 2500 * 10 = 25,000 ops. Very fast.
                
                new_order = current_order[:i] + current_order[i+1:]
                if i < j:
                    # Index j shifts left by 1 in the reduced list?
                    # Original: [0, 1, 2, 3, 4], i=1, j=3.
                    # Remove 1: [0, 2, 3, 4].
                    # Insert at 3: [0, 2, 3, job1, 4].
                    # So in the reduced list, we insert at index j?
                    # Let's trace:
                    # new_order is length N-1.
                    # We want the job to end up at index j in the FINAL list of length N.
                    # If we insert at pos k in the reduced list, the final index is k if k < i, or k+1 if k >= i?
                    # Actually, easier:
                    # Just construct the list directly.
                    pass
                
                # Simpler construction:
                temp_order = current_order.copy()
                job = temp_order.pop(i)
                if i < j:
                    # The index j in the original list corresponds to j-1 in the popped list?
                    # Example: [A, B, C, D], i=1(B), j=3(D).
                    # Pop B: [A, C, D].
                    # We want B to be at index 3 in [A, B, C, D] -> [A, C, D, B]? No.
                    # Original indices: 0:A, 1:B, 2:C, 3:D.
                    # Move B to pos 3.
                    # Result: [A, C, D, B].
                    # In popped list [A, C, D], B is inserted at end (index 3).
                    # So if i < j, insert at j in the popped list?
                    # Popped list length is N-1. Max index is N-2.
                    # If j = N-1, insert at N-1 (end).
                    # So yes, if i < j, insert at j.
                    insert_idx = j
                else:
                    # i > j
                    # Example: [A, B, C, D], i=3(D), j=1(B).
                    # Pop D: [A, B, C].
                    # We want D to be at index 1.
                    # Result: [A, D, B, C].
                    # In popped list [A, B, C], insert at 1.
                    # So if i > j, insert at j.
                    insert_idx = j
                
                new_order = temp_order[:insert_idx] + [job] + temp_order[insert_idx:]
                
                ms = _makespan(processing_times, new_order)
                
                # Aspiration Criterion: Accept if it improves global best
                if ms < best_ms:
                    is_tabu = False
                    
                if ms < best_move_ms:
                    best_move_ms = ms
                    best_move_job = job
                    best_move_from = i
                    best_move_to = j
                    best_is_tabu = is_tabu
        
        # Apply Best Move
        if best_move_job != -1:
            # If the best move is tabu and doesn't meet aspiration, we should pick the best NON-TABU move.
            # My loop above picked the absolute best, but flagged if it was tabu.
            # If it's tabu, we need to find the best non-tabu one.
            # To avoid a second full pass, I will modify the selection logic:
            # Keep track of best_non_tabu and best_tabu.
            
            # Re-evaluating selection logic for correctness:
            # I'll do a quick re-scan or store both during the loop.
            # Let's refine the loop above to store best_non_tabu.
            
            # For now, if best is tabu, we might skip it or accept it if we allow cycling?
            # Standard Tabu: Must accept best non-tabu.
            
            # Let's re-run the selection logic properly in a separate pass if needed, 
            # or just accept the best if it's not tabu.
            # Given the time constraints, I will implement a simpler logic:
            # If best_is_tabu, find the best non-tabu move.
            
            if best_is_tabu:
                # Find best non-tabu
                found = False
                for i in range(n):
                    if found: break
                    job_idx = current_order[i]
                    for j in range(n):
                        if i == j: continue
                        key = (job_idx, j)
                        if key in tabu_list and tabu_list[key] > iteration:
                            continue
                        # This is not tabu
                        # Calc MS
                        temp_order = current_order.copy()
                        job = temp_order.pop(i)
                        insert_idx = j
                        new_order = temp_order[:insert_idx] + [job] + temp_order[insert_idx:]
                        ms = _makespan(processing_times, new_order)
                        
                        if ms < best_move_ms: # Wait, best_move_ms is the tabu one. We want best NON-TABU.
                             # We need a separate variable best_non_tabu_ms
                             pass
                        # This double pass is expensive. 
                        # Alternative: In the first loop, track best_non_tabu.
                        
            # Let's restart the neighborhood search with proper tracking to avoid this bug.
            
            best_move_ms = float('inf')
            best_non_tabu_ms = float('inf')
            best_move_job = -1
            best_move_from = -1
            best_move_to = -1
            best_non_tabu_job = -1
            best_non_tabu_from = -1
            best_non_tabu_to = -1
            
            for i in range(n):
                if time.time() > deadline - 5.0: break
                job_idx = current_order[i]
                for j in range(n):
                    if i == j: continue
                    key = (job_idx, j)
                    is_tabu = (key in tabu_list and tabu_list[key] > iteration)
                    
                    temp_order = current_order.copy()
                    job = temp_order.pop(i)
                    insert_idx = j
                    new_order = temp_order[:insert_idx] + [job] + temp_order[insert_idx:]
                    ms = _makespan(processing_times, new_order)
                    
                    if ms < best_move_ms:
                        best_move_ms = ms
                        best_move_job = job
                        best_move_from = i
                        best_move_to = j
                    
                    if not is_tabu and ms < best_non_tabu_ms:
                        best_non_tabu_ms = ms
                        best_non_tabu_job = job
                        best_non_tabu_from = i
                        best_non_tabu_to = j
                        
            # Decision
            if best_non_tabu_job != -1:
                # Use best non-tabu
                final_job = best_non_tabu_job
                final_from = best_non_tabu_from
                final_to = best_non_tabu_to
                final_ms = best_non_tabu_ms
            else:
                # All moves are tabu? Use best overall (aspiration or forced)
                final_job = best_move_job
                final_from = best_move_from
                final_to = best_move_to
                final_ms = best_move_ms
            
            if final_job != -1:
                # Apply move
                temp_order = current_order.copy()
                job = temp_order.pop(final_from)
                insert_idx = final_to
                current_order = temp_order[:insert_idx] + [job] + temp_order[insert_idx:]
                current_ms = final_ms
                
                # Update Tabu List
                # Tabu the destination
                tabu_list[(final_job, final_to)] = iteration + tabu_tenure
                # Optionally tabu the source to prevent immediate undo
                # tabu_list[(final_job, final_from)] = iteration + tabu_tenure
                
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_order = list(current_order)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            break # No moves found?

    return best_order
# EVOLVE-BLOCK-END
