"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    c_prev = [0.0] * n_machines
    c_curr = [0.0] * n_machines
    
    for i in range(n_jobs):
        job_idx = order[i]
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            p = p_row[j]
            if i == 0 and j == 0:
                c_curr[j] = p
            elif i == 0:
                c_curr[j] = c_curr[j-1] + p
            elif j == 0:
                c_curr[j] = c_prev[j] + p
            else:
                c_curr[j] = max(c_prev[j], c_curr[j-1]) + p
        c_prev, c_curr = c_curr, c_prev
    
    return c_prev[-1]

def _makespan_with_c(processing_times, order):
    """Compute makespan and return the full C matrix."""
    if len(order) == 0:
        return 0.0, None
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    C = [[0.0] * n_machines for _ in range(n_jobs)]
    
    for i in range(n_jobs):
        job_idx = order[i]
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            p = p_row[j]
            if i == 0 and j == 0:
                C[i][j] = p
            elif i == 0:
                C[i][j] = C[i][j-1] + p
            elif j == 0:
                C[i][j] = C[i-1][j] + p
            else:
                C[i][j] = max(C[i-1][j], C[i][j-1]) + p
    
    return C[n_jobs-1][n_machines-1], C

def _neh_rule(processing_times):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    order = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            candidate = order[:pos] + [job] + order[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order.insert(best_pos, job)
    
    return order

def _local_search_insertion(processing_times, order, C, time_limit=10.0):
    """
    Local search using insertion moves with incremental makespan evaluation.
    """
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    if n_jobs <= 1:
        return order, C, C[n_jobs-1][n_machines-1] if n_jobs > 0 else 0.0
    
    current_order = list(order)
    current_C = [row[:] for row in C]
    current_ms = current_C[n_jobs-1][n_machines-1]
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        # Try all insertion moves
        # src: job to move, dst: position to insert at
        for src in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            job = current_order[src]
            for dst in range(n_jobs):
                if time.time() - start_time > time_limit:
                    break
                if src == dst:
                    continue
                
                # Determine the range of rows to recalculate
                # The job moves from src to dst.
                # If src < dst: job moves right. Rows src to dst are affected.
                # If src > dst: job moves left. Rows dst to src are affected.
                lo = min(src, dst)
                hi = max(src, dst)
                
                # We need to simulate the new order for the segment [lo, hi]
                # and recalculate C for these rows.
                # The row before 'lo' (if exists) provides the starting boundary.
                
                # Create a temporary list for the new order segment
                # Original segment: current_order[lo : hi+1]
                # Remove job at src, insert at dst (adjusted for shift)
                
                # It's safer to construct the new order for the whole array? No, too slow.
                # Let's construct the new segment.
                
                # Get the jobs in the segment
                segment_jobs = current_order[lo : hi + 1]
                
                # Remove the job at relative position src - lo
                rel_src = src - lo
                job_to_move = segment_jobs[rel_src]
                segment_jobs.pop(rel_src)
                
                # Calculate the new relative destination
                # If src < dst, the destination index in the new list is dst - lo
                # If src > dst, the destination index in the new list is dst - lo
                # Wait, if we pop an element before dst, dst shifts left by 1?
                # If src < dst:
                #   Original indices: ... src ... dst ...
                #   Pop src. Elements shift left.
                #   New index for dst is dst - 1.
                #   We want to insert at position dst - 1?
                #   Let's check:
                #   Order: A B C D E. src=1 (B), dst=3 (D).
                #   Segment: B C D. Pop B -> C D.
                #   Insert B at dst-1 = 2? C D B?
                #   Original: A B C D E -> A C D B E.
                #   Yes.
                #   If src > dst:
                #   Order: A B C D E. src=3 (D), dst=1 (B).
                #   Segment: B C D. Pop D (rel 2) -> B C.
                #   Insert D at dst-lo = 1-0 = 1? B D C.
                #   Original: A B C D E -> A B D C E.
                #   Yes.
                
                if src < dst:
                    insert_idx = dst - lo
                else:
                    insert_idx = dst - lo
                
                segment_jobs.insert(insert_idx, job_to_move)
                
                # Now recalculate C for rows lo to hi using segment_jobs
                # We need C[lo-1] as the previous row completion times
                if lo == 0:
                    prev_row = [0.0] * n_machines
                else:
                    prev_row = current_C[lo - 1]
                
                # Simulate the calculation for the segment
                # new_C_segment is a list of lists
                # We only care about the final makespan, which is at the end of the whole schedule.
                # However, the completion times of the last row of the segment (hi) will affect subsequent rows (hi+1 to n_jobs-1).
                # So we must recalculate from lo to n_jobs-1.
                
                # Optimization: If hi == n_jobs - 1, we only need to recalculate up to hi.
                # If hi < n_jobs - 1, we must recalculate up to n_jobs - 1.
                
                # Let's just recalculate from lo to n_jobs - 1.
                # This is O((n_jobs - lo) * M).
                
                # Prepare a temporary C array for the affected part
                # We can just overwrite a copy of the current C?
                # To avoid allocating a full C matrix every time, we can use a smaller buffer or just calculate the new makespan directly without storing the full C,
                # but we need to update current_C if the move is accepted.
                
                # Let's calculate the new makespan first.
                # We need the completion times of row lo-1.
                # Then we iterate through the new order from lo to n_jobs-1.
                
                # Construct the new full order? No, just the tail.
                # The tail of the order is:
                # 1. The segment_jobs (which corresponds to positions lo to hi)
                # 2. The rest of the order (positions hi+1 to n_jobs-1)
                
                # Wait, the jobs in the segment are exactly the jobs from lo to hi.
                # The jobs after hi are unchanged.
                # So the new order from lo onwards is:
                # segment_jobs + current_order[hi+1:]
                
                new_tail_jobs = segment_jobs + current_order[hi+1:]
                
                # Now calculate makespan for this new tail, starting from prev_row
                c_prev = prev_row[:]
                c_curr = [0.0] * n_machines
                
                # We need to iterate through new_tail_jobs
                # The number of jobs is len(new_tail_jobs)
                # We need to know which "row index" we are at to handle the i==0 case?
                # No, the recurrence is local.
                # C[k][j] depends on C[k-1][j] and C[k][j-1].
                # We have c_prev (which is C[lo-1] if lo>0, or zeros if lo==0).
                # We just iterate.
                
                for i, job_idx in enumerate(new_tail_jobs):
                    p_row = processing_times[job_idx]
                    for j in range(n_machines):
                        p = p_row[j]
                        if i == 0 and j == 0:
                            # This case only happens if lo == 0
                            c_curr[j] = p
                        elif i == 0:
                            # First job in the tail (which is job at position lo)
                            # If lo == 0, c_prev is zeros.
                            # C[0][j] = C[0][j-1] + p
                            c_curr[j] = c_curr[j-1] + p
                        elif j == 0:
                            # First machine for subsequent jobs
                            # C[k][0] = C[k-1][0] + p
                            c_curr[j] = c_prev[j] + p
                        else:
                            # C[k][j] = max(C[k-1][j], C[k][j-1]) + p
                            c_curr[j] = max(c_prev[j], c_curr[j-1]) + p
                    c_prev, c_curr = c_curr, c_prev
                
                new_ms = c_prev[-1]
                
                if new_ms < current_ms:
                    # Accept the move
                    # Update current_order
                    current_order[lo : hi + 1] = segment_jobs
                    # Update current_C
                    # We need to recalculate C for rows lo to n_jobs-1 and store them.
                    # We can reuse the calculation above but store the rows.
                    
                    # Re-run the calculation to fill current_C
                    if lo == 0:
                        prev_row = [0.0] * n_machines
                    else:
                        prev_row = current_C[lo - 1]
                        
                    c_prev = prev_row[:]
                    c_curr = [0.0] * n_machines
                    
                    for i, job_idx in enumerate(new_tail_jobs):
                        p_row = processing_times[job_idx]
                        row_vals = [0.0] * n_machines
                        for j in range(n_machines):
                            p = p_row[j]
                            if i == 0 and j == 0:
                                row_vals[j] = p
                            elif i == 0:
                                row_vals[j] = row_vals[j-1] + p
                            elif j == 0:
                                row_vals[j] = c_prev[j] + p
                            else:
                                row_vals[j] = max(c_prev[j], row_vals[j-1]) + p
                        current_C[lo + i] = row_vals
                        c_prev = row_vals[:]
                    
                    current_ms = new_ms
                    improved = True
                    # Restart the loop? Or continue?
                    # Continuing is fine, but the order has changed.
                    # The indices src and dst are no longer valid for the next iteration of the inner loop?
                    # Actually, we are in a for loop over src and dst.
                    # If we change the order, the job at `src` is no longer `job`.
                    # It is safer to break out of the inner loop and restart the outer loop?
                    # Or just break out of both and restart the while loop?
                    # Breaking out of the while loop and restarting is cleanest.
                    break
            if improved:
                break
                
    return current_order, current_C, current_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Generate initial solution using NEH
    initial_order = _neh_rule(processing_times)
    initial_ms, initial_C = _makespan_with_c(processing_times, initial_order)
    
    # Run local search with time budget
    remaining_time = deadline - time.time()
    if remaining_time < 5.0:
        return initial_order
    
    # Use most of the remaining time for local search
    time_limit = remaining_time * 0.9
    best_order, best_C, best_ms = _local_search_insertion(
        processing_times, initial_order, initial_C, 
        time_limit=time_limit
    )
    
    # If we have significant time left, try a few restarts
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        best_overall = best_order
        best_overall_ms = best_ms
        
        # Try a few random restarts with short time limits
        num_restarts = min(int(remaining_time / 10.0), 5)
        for _ in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Random initial order
            random_order = list(range(n_jobs))
            random.shuffle(random_order)
            
            # Quick local search with shorter time limit
            tl = min(remaining_time - 5.0, 10.0)
            if tl <= 0:
                break
            
            # Get C for random order
            _, rand_C = _makespan_with_c(processing_times, random_order)
            
            order, C, ms = _local_search_insertion(
                processing_times, random_order, rand_C,
                time_limit=tl
            )
            
            if ms < best_overall_ms:
                best_overall = order
                best_overall_ms = ms
            
            remaining_time = deadline - time.time()
            if remaining_time <= 5.0:
                break
        
        return best_overall
    
    return best_order
# EVOLVE-BLOCK-END
