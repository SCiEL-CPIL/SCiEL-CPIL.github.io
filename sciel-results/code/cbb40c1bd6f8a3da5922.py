"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan_full(perm, pt):
    """
    Calculate makespan from scratch.
    O(n*m)
    """
    if len(perm) == 0:
        return 0.0
    m = pt.shape[1]
    c = [0.0] * m
    for job_idx in perm:
        prev_c = 0.0
        row = pt[job_idx]
        for k in range(m):
            val = row[k]
            if c[k] > prev_c:
                c[k] = c[k] + val
            else:
                c[k] = prev_c + val
            prev_c = c[k]
    return c[-1]

def _compute_prefix_completions(perm, pt):
    """
    Compute the completion times for each job in the sequence.
    Returns a list of lists: prefix_completions[i] is the list of completion times 
    of job perm[i] on each machine.
    """
    n = len(perm)
    m = pt.shape[1]
    prefix_completions = []
    c = [0.0] * m
    for i in range(n):
        job_idx = perm[i]
        row = pt[job_idx]
        prev_c = 0.0
        job_completions = [0.0] * m
        for k in range(m):
            val = row[k]
            if c[k] > prev_c:
                c[k] = c[k] + val
            else:
                c[k] = prev_c + val
            prev_c = c[k]
            job_completions[k] = c[k]
        prefix_completions.append(job_completions)
    return prefix_completions

def _makespan_delta(perm, pt, prefix_completions, i, j):
    """
    Calculate the new makespan if job at position i is moved to position j.
    Assumes i != j.
    Returns (new_makespan, new_prefix_completions)
    """
    n = len(perm)
    m = pt.shape[1]
    
    # Extract job
    job_idx = perm[i]
    row = pt[job_idx]
    
    # Determine the range of affected jobs
    # If i < j, the job is moved right. Jobs i+1...j shift left.
    # If i > j, the job is moved left. Jobs j...i-1 shift right.
    
    # We need to recalculate completions for the affected segment.
    # The segment is [min(i,j), max(i,j)] inclusive, plus the moved job.
    
    # Let's create a new list of completions to avoid modifying the original one in-place 
    # until we are sure it's better, or we can modify in place and revert if not.
    # For speed, we'll compute the new completions for the affected range into a temp list.
    
    # The jobs before min(i,j) are unchanged.
    # The job after max(i,j) depends on the job at max(i,j) (which might be the moved job or a shifted one).
    # Actually, the completion of the last job in the affected range determines the completion of the job after it.
    
    start = min(i, j)
    end = max(i, j)
    
    # The job to move
    job_to_move = perm[i]
    row_to_move = pt[job_to_move]
    
    # The sequence of jobs in the range [start, end] excluding the moved job, 
    # but we need to simulate the shift.
    
    # Let's identify the jobs that stay in the "gap"
    # If moving i -> j:
    # Original: ... A [J_i] B ... C [J_j] D ...
    # If i < j: ... A B ... C [J_i] D ...
    # If i > j: ... A [J_i] B ... C D ...
    
    # We need to recalculate completions for positions start to end.
    # The completion time of the job before `start` is prefix_completions[start-1] (if start > 0).
    
    prev_completions = prefix_completions[start-1] if start > 0 else [0.0]*m
    
    # Build the list of jobs in the affected range in their new order
    # The moved job is removed from i.
    # The jobs from start to end (excluding i) are shifted.
    
    # Let's just construct the new sequence for the range [start, end]
    # The length of the range is (end - start + 1)
    # The new sequence in this range will have the same jobs, but J_i is at position j relative to start?
    # No, J_i is at index j in the global perm.
    
    # Let's iterate through the new positions in the range [start, end]
    # New position p in [start, end]:
    # If p == j, the job is J_i.
    # Else, the job is the one that was originally at some other index.
    
    # It's easier to think:
    # The set of jobs in the range [start, end] is fixed.
    # The new order is:
    # If i < j:
    #   Positions start...i-1: same as before
    #   Position i: job that was at i+1
    #   ...
    #   Position j-1: job that was at j
    #   Position j: job that was at i (the moved job)
    # If i > j:
    #   Position start...j-1: same as before
    #   Position j: job that was at i (the moved job)
    #   Position j+1: job that was at j
    #   ...
    #   Position i: job that was at i-1
    
    # Let's compute the new completions for the range.
    new_range_completions = []
    
    # We need to know the job index for each new position p in [start, end]
    for p in range(start, end + 1):
        if p == j:
            current_job = job_to_move
        else:
            # Determine which original job index corresponds to this new position
            if i < j:
                # Moving right
                if p < i:
                    # Unchanged
                    orig_idx = p
                elif p < j:
                    # Shifted left: job at p+1 moves to p
                    orig_idx = p + 1
                else:
                    # p == j is handled above, so this shouldn't happen for p > j in this branch?
                    # Wait, if i < j, p goes up to j.
                    # If p > i and p < j, orig_idx = p + 1.
                    # If p == j, it's the moved job.
                    # What if p > j? The loop ends at end=j. So p <= j.
                    orig_idx = p + 1
            else:
                # Moving left (i > j)
                if p < j:
                    # Unchanged
                    orig_idx = p
                elif p > j:
                    # Shifted right: job at p-1 moves to p
                    orig_idx = p - 1
                else:
                    # p == j is handled above
                    orig_idx = p - 1 # This case is p > j, but if p==j it's moved job.
                    # Wait, if p > j, then p is in (j, i].
                    # The job at p was originally at p-1?
                    # Let's trace: i=5, j=2.
                    # Pos 2: Job 5.
                    # Pos 3: Job 2.
                    # Pos 4: Job 3.
                    # Pos 5: Job 4.
                    # So for p=3, orig_idx=2. p=4, orig_idx=3. p=5, orig_idx=4.
                    # So orig_idx = p - 1 is correct for p > j.
        
        # Now we have the job index for position p
        # We need to calculate its completion times based on prev_completions
        # But prev_completions changes as we go!
        
        # So we must update prev_completions as we iterate.
        # We can't do this in a single pass if we just look up orig_idx.
        # We need to simulate the flow.
        
        pass # This logic is getting complex inside the loop. Let's restructure.
        
    # Better approach:
    # 1. Copy the completions for the range into a temporary list.
    # 2. Update them sequentially.
    
    # Let's just re-evaluate the entire range [start, end] from scratch using the jobs in their new order.
    
    # Determine the sequence of jobs in the new range
    new_jobs_in_range = []
    for p in range(start, end + 1):
        if p == j:
            new_jobs_in_range.append(job_to_move)
        else:
            if i < j:
                if p < i:
                    new_jobs_in_range.append(perm[p])
                else: # p > i and p < j (since p <= end=j and p!=j)
                    new_jobs_in_range.append(perm[p+1])
            else: # i > j
                if p < j:
                    new_jobs_in_range.append(perm[p])
                else: # p > j and p <= end=i
                    new_jobs_in_range.append(perm[p-1])
                    
    # Now simulate the flow for these jobs
    # Initial completion times for the machine at the start of the range
    # This is the completion times of the job at position start-1
    c = list(prev_completions)
    
    new_range_comps = []
    for job in new_jobs_in_range:
        row = pt[job]
        prev_c = 0.0
        job_comps = [0.0] * m
        for k in range(m):
            val = row[k]
            if c[k] > prev_c:
                c[k] = c[k] + val
            else:
                c[k] = prev_c + val
            prev_c = c[k]
            job_comps[k] = c[k]
        new_range_comps.append(job_comps)
        
    # The makespan is the max of the last completion time of the last job in the range
    # BUT, we must consider if the last job in the range is NOT the last job in the whole permutation.
    # If end < n-1, the job after `end` will add to its processing time.
    # However, the makespan is determined by the last job in the sequence.
    # If the moved job is not the last job, the completion of the last job in the range
    # determines the start of the next job.
    
    # The new completion times for the range are in new_range_comps.
    # The last job in the range is at index end.
    last_job_comp = new_range_comps[-1]
    
    # If end == n-1, the makespan is max(last_job_comp)
    # If end < n-1, the makespan is determined by the job at n-1.
    # The job at n-1 starts after the job at n-2 finishes.
    # The completion of the job at n-1 in the NEW sequence depends on the completion of the job at n-2.
    # If n-2 >= end, then the job at n-2 is in the range, and its new completion is in new_range_comps.
    # If n-2 < end, then the job at n-2 is unchanged, and its completion is in prefix_completions.
    
    # Actually, the makespan is the completion time of the last job in the permutation.
    # Let's find the completion time of the last job (index n-1) in the new sequence.
    
    if end == n - 1:
        new_makespan = max(last_job_comp)
    else:
        # The last job in the permutation is at index n-1.
        # It is NOT in the range [start, end] because end < n-1.
        # So the job at n-1 is unchanged in terms of identity, but its start time depends on the job at n-2.
        # The job at n-2 might be in the range or not.
        
        # Let's trace the completion times from `end` to `n-1`.
        # We have the new completion times for jobs in [start, end].
        # The job at `end` has new completion `last_job_comp`.
        
        # The job at `end+1` is unchanged (identity), but its start time depends on `end`.
        # We need to recalculate completions for jobs end+1 to n-1.
        
        # This is O((n-end)*m). If the move is near the end, this is small.
        # If the move is near the start, this is large.
        
        # Optimization: We only need to recalculate if the completion of the job at `end`
        # has changed in a way that affects the downstream jobs.
        # Since we are calculating the exact makespan, we must propagate.
        
        c_downstream = list(last_job_comp)
        
        for p in range(end + 1, n):
            job = perm[p] # Identity is unchanged for p > end
            row = pt[job]
            prev_c = 0.0
            job_comps = [0.0] * m
            for k in range(m):
                val = row[k]
                if c_downstream[k] > prev_c:
                    c_downstream[k] = c_downstream[k] + val
                else:
                    c_downstream[k] = prev_c + val
                prev_c = c_downstream[k]
                job_comps[k] = c_downstream[k]
            
            if p == n - 1:
                new_makespan = max(job_comps)
            else:
                # Continue propagation
                # c_downstream is already updated to the completions of job p
                pass
        
    return new_makespan

def _neh(processing_times):
    """NEH heuristic: build solution by inserting jobs in order of decreasing total processing time."""
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n)), 0.0
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = []
    for job in order:
        best_pos = -1
        best_ms = float('inf')
        for k in range(len(perm) + 1):
            trial = perm[:k] + [job] + perm[k:]
            ms = _makespan_full(trial, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = k
        perm.insert(best_pos, job)
    
    return perm, _makespan_full(perm, processing_times)

def _insertion_local_search(perm, pt, max_iters=500):
    """
    Insertion-based local search using delta evaluation.
    """
    n = len(perm)
    if n <= 1:
        return list(perm), _makespan_full(perm, pt)
    
    current_perm = list(perm)
    current_ms = _makespan_full(current_perm, pt)
    prefix_comps = _compute_prefix_completions(current_perm, pt)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n):
            job = current_perm[i]
            # Try inserting at each position k != i
            for k in range(n - 1):
                if k == i:
                    continue
                # Calculate new makespan using delta
                new_ms = _makespan_delta(current_perm, pt, prefix_comps, i, k)
                if new_ms < current_ms:
                    # Accept move
                    # Update perm
                    current_perm.pop(i)
                    current_perm.insert(k, job)
                    current_ms = new_ms
                    # Recalculate prefix completions (O(n*m) cost, but only on improvement)
                    prefix_comps = _compute_prefix_completions(current_perm, pt)
                    improved = True
                    break
            if improved:
                break
    
    return current_perm, current_ms

def _or_opt_move(perm, n, rng):
    """
    Generate an or-opt move: remove a segment of length 1, 2, or 3 and reinsert elsewhere.
    Returns (new_perm, move_key)
    """
    # Choose segment length: 1 (50%), 2 (30%), 3 (20%)
    r = rng.random()
    if r < 0.5:
        seg_len = 1
    elif r < 0.8:
        seg_len = 2
    else:
        seg_len = 3
    
    if n <= seg_len:
        seg_len = max(1, n - 1)
    
    # Choose start position for segment
    start = rng.randint(0, n - seg_len)
    
    # Choose insertion position (not the same as original)
    insert_pos = rng.randint(0, n - seg_len)
    while insert_pos == start:
        insert_pos = rng.randint(0, n - seg_len)
    
    # Extract segment
    segment = perm[start:start + seg_len]
    
    # Build new permutation
    without_segment = perm[:start] + perm[start + seg_len:]
    
    # Adjust insertion position if it was after the removed segment
    if insert_pos > start:
        insert_pos -= seg_len
    
    new_perm = without_segment[:insert_pos] + segment + without_segment[insert_pos:]
    
    # Move key for tabu: (start, insert_pos, seg_len)
    move_key = (start, insert_pos, seg_len)
    
    return new_perm, move_key

def _tabu_search(perm, pt, rng, time_limit):
    """
    Tabu search with or-opt moves and delta evaluation.
    """
    n = len(perm)
    if n <= 1:
        return list(perm), _makespan_full(perm, pt)
    
    current_perm = list(perm)
    current_ms = _makespan_full(current_perm, pt)
    prefix_comps = _compute_prefix_completions(current_perm, pt)
    
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # Tabu list: store move keys for tabu_tenure iterations
    tabu_tenure = max(10, n // 2)
    tabu_list = []
    
    # Temperature for accepting worse moves
    T = best_ms * 0.02
    T_min = best_ms * 0.0005
    alpha = 0.999
    
    start_time = time.time()
    stall_count = 0
    max_stall = 1000 # Increased stall limit as delta eval is faster
    
    while time.time() < time_limit:
        if T < T_min:
            break
        
        # Generate a neighbor
        new_perm, move_key = _or_opt_move(current_perm, n, rng)
        
        # Calculate new makespan
        # For or-opt moves, delta evaluation is more complex than single job insertion.
        # We can use _makespan_full for or-opt moves if n is small, or implement a specific delta.
        # Given the complexity of or-opt delta, and that n is likely small (<50), 
        # _makespan_full is O(n*m). If n=50, m=5, that's 250 ops.
        # Delta insertion was O(m) + O((n-end)m).
        # Let's stick to _makespan_full for or-opt to keep code simple, 
        # but rely on the speedup from insertion LS to do more restarts/polish.
        # Actually, we can just use _makespan_full here. The bottleneck was the inner loop of LS.
        
        new_ms = _makespan_full(new_perm, pt)
        
        # Check tabu status
        is_tabu = move_key in tabu_list
        
        # Aspiration: accept if it improves best
        aspiration = new_ms < best_ms
        
        if new_ms < current_ms or (not is_tabu and (aspiration or rng.random() < np.exp(-(new_ms - current_ms) / T))):
            # Accept the move
            current_perm = new_perm
            current_ms = new_ms
            prefix_comps = _compute_prefix_completions(current_perm, pt) # Update for potential future use? Not needed for or-opt eval, but good for consistency if we switch.
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = list(current_perm)
                stall_count = 0
            else:
                stall_count += 1
            
            # Update tabu list
            tabu_list.append(move_key)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
        else:
            stall_count += 1
        
        T *= alpha
        
        if stall_count > max_stall:
            break
    
    return best_perm, best_ms

def order_jobs(processing_times):
    start_time = time.time()
    # 120s limit, use 110s to be safe
    deadline = start_time + 110.0
    
    n, m = processing_times.shape
    
    if n <= 1:
        return list(range(n))
    
    # 1. Initial Solution: NEH with insertion
    best_perm, best_ms = _neh(processing_times)
    
    # Apply insertion local search to the NEH solution
    best_perm, best_ms = _insertion_local_search(best_perm, processing_times, max_iters=200)
    
    # 2. Multi-restart Tabu Search with or-opt moves
    rng = np.random.RandomState(42)
    
    max_restarts = 10 # Can afford more restarts now
    for restart in range(max_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Perturb the best solution to create a diverse starting point
        current_perm = list(best_perm)
        
        # Use or-opt perturbation for more effective diversification
        num_perturbs = max(2, n // 5)
        for _ in range(num_perturbs):
            if n > 1:
                perturbed_perm, _ = _or_opt_move(current_perm, n, rng)
                current_perm = perturbed_perm
        
        current_ms = _makespan_full(current_perm, processing_times)
        
        # Run tabu search
        tabu_start = time.time()
        tabu_time_limit = deadline - 20.0
        ts_perm, ts_ms = _tabu_search(current_perm, processing_times, rng, tabu_time_limit)
        
        if ts_ms < best_ms:
            best_ms = ts_ms
            best_perm = ts_perm
        
        # Apply insertion local search to the tabu search result if it's reasonably close
        if current_ms <= best_ms * 1.1 and time.time() < deadline - 10.0:
            polished_perm, polished_ms = _insertion_local_search(current_perm, processing_times, max_iters=150)
            if polished_ms < best_ms:
                best_ms = polished_ms
                best_perm = polished_perm
        
        # Also polish the tabu result
        if time.time() < deadline - 10.0:
            polished_perm, polished_ms = _insertion_local_search(ts_perm, processing_times, max_iters=150)
            if polished_ms < best_ms:
                best_ms = polished_ms
                best_perm = polished_perm
    
    # Final polish with more iterations if time allows
    if time.time() < deadline - 3.0:
        polished_perm, polished_ms = _insertion_local_search(best_perm, processing_times, max_iters=300)
        if polished_ms < best_ms:
            best_ms = polished_ms
            best_perm = polished_perm
    
    return list(best_perm)
# EVOLVE-BLOCK-END
