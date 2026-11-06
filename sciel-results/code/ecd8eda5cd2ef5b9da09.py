"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _calc_makespan(processing_times, perm):
    """Calculate makespan for a given permutation."""
    n_machines = processing_times.shape[1]
    if n_machines == 1:
        return float(processing_times[perm].sum())
    completion = [0.0] * n_machines
    for job in perm:
        row = processing_times[job]
        for m in range(n_machines):
            if m == 0:
                completion[0] += row[m]
            else:
                c_prev = completion[m - 1]
                c_curr = completion[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                completion[m] = c_curr + row[m]
    return completion[-1]


def _calc_makespan_incremental(processing_times, perm, pos, job, completion):
    """
    Calculate the new makespan after inserting job at position pos.
    Returns (new_makespan, new_completion) or (inf, None) if not valid.
    """
    n_machines = processing_times.shape[1]
    row = processing_times[job]
    
    # We need to compute the completion times for the new sequence.
    # Instead of full recompute, we can be smart, but for simplicity and correctness
    # in the insertion neighborhood, we often just recompute the tail.
    # However, for Tabu Search efficiency, let's try a faster check.
    
    # Actually, for a robust implementation, let's stick to a slightly optimized full check
    # or use the property that only positions >= pos are affected.
    
    # Let's compute the completion times for the new sequence.
    # The sequence is: perm[0:pos] + [job] + perm[pos:]
    
    # We need the completion time at the end of perm[0:pos-1] to start the chain for the new job.
    # If pos == 0, start from 0.
    # If pos > 0, start from completion[pos-1] of the ORIGINAL sequence?
    # No, because the jobs after pos-1 are shifted.
    
    # Let's just do a fast full calculation for the tail.
    # Actually, a full calculation is O(n*m). With n~100, m~10, it's 1000 ops.
    # Tabu search might do 1000s of moves. 1000 * 1000 = 1M ops. That's fine.
    
    # Let's just call _calc_makespan for simplicity in the Tabu loop, 
    # but we can optimize by pre-computing prefix completions.
    
    # For now, let's use a helper that computes makespan given a list.
    # To speed up, we can pass the processing times array directly.
    
    # Let's implement a fast version that takes the permutation list.
    # We will use this in the main loop.
    
    # To make Tabu Search fast, we need to evaluate many neighbors.
    # Let's write a specialized function for insertion move evaluation.
    
    # New sequence: [p0, ..., p_{pos-1}, job, p_pos, ..., p_{n-1}]
    # Completion at pos-1 is C_{pos-1} from the original sequence.
    # Let C_orig be the completion times of the original sequence.
    
    # If pos == 0:
    #   C_new[0] = row[0]
    #   C_new[m] = max(C_new[m-1], 0) + row[m] for m>0? No, C_new[m] = max(C_new[m-1], C_orig[m-1]?) No.
    #   The machine m completion time for the new job depends on the previous job on machine m.
    #   The previous job on machine m in the new sequence is:
    #   - If m=0: the job at pos-1 in the new sequence (which is p_{pos-1} if pos>0, else none).
    #   - If m>0: the job at pos-1 in the new sequence is p_{pos-1}. But the completion time of p_{pos-1} on machine m is C_orig[pos-1, m].
    #   Wait, the completion time of p_{pos-1} on machine m is fixed.
    #   So C_new_job[m] = max(C_new_job[m-1], C_orig[pos-1, m]) + row[m]
    
    #   Then for the subsequent jobs p_pos, p_{pos+1}, ...:
    #   Their start times are delayed by the new job.
    #   C_new[k, m] for k >= pos in the new sequence (which corresponds to p_{pos + k - pos} in original? No.)
    #   The job at index k in the new sequence (k >= pos+1) is p_{k-1} from the original sequence (shifted by 1).
    #   This is getting complicated.
    
    # Let's stick to the full calculation for correctness first, and optimize if needed.
    # Given the time budget (150s), and typical n=50-100, full calc is fast enough.
    
    new_perm = list(perm[:pos]) + [job] + list(perm[pos:])
    return _calc_makespan(processing_times, new_perm)


def _v_shaped_heuristic(processing_times):
    """
    V-shaped heuristic for Flow Shop.
    Jobs are sorted by total processing time.
    The job with the largest total processing time is placed first.
    Then, the next largest is placed either at the beginning or the end,
    whichever yields the smaller partial makespan.
    """
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        
        # Try inserting at the beginning
        seq_start = [job] + sequence
        ms_start = _calc_makespan(processing_times, seq_start)
        
        # Try inserting at the end
        seq_end = sequence + [job]
        ms_end = _calc_makespan(processing_times, seq_end)
        
        if ms_start <= ms_end:
            sequence = seq_start
        else:
            sequence = seq_end
            
    return sequence


def _neh_heuristic(processing_times):
    """
    NEH heuristic: Sort jobs by total processing time descending,
    then insert each job into the position that minimizes makespan.
    """
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(sequence) + 1):
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            ms = _calc_makespan(processing_times, new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence


def _random_permutation(n_jobs):
    """Generate a random permutation of job indices."""
    perm = list(range(n_jobs))
    random.shuffle(perm)
    return perm


def _tabu_search(processing_times, sequence, deadline):
    """
    Tabu Search with insertion neighborhood.
    Uses a dynamic tabu list to prevent cycling.
    """
    n_jobs = len(sequence)
    if n_jobs < 2:
        return list(sequence), _calc_makespan(processing_times, sequence)
    
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    # Tabu list: stores (job, pos) that were last moved
    # We use a dictionary to store the iteration number when a move was made.
    # A move (i, j) is tabu if the last move involving job i to position j was within the last `tabu_tenure` iterations.
    # Simpler: store the last position each job was moved from and to.
    # Let's use a list of (job, from_pos, to_pos) with timestamps.
    # Or even simpler: a set of (job, to_pos) that are tabu.
    # We will use a list of (job, to_pos) and their expiration time.
    
    tabu_list = []  # list of (job, to_pos, expiration_iter)
    tabu_tenure = 7  # Dynamic tenure
    
    no_improve_count = 0
    max_no_improve = 50
    iteration = 0
    
    while time.time() < deadline:
        iteration += 1
        
        # Dynamic tabu tenure
        if no_improve_count > 30:
            tabu_tenure = 10
        else:
            tabu_tenure = 7
            
        best_move = None
        best_move_makespan = float('inf')
        best_move_is_tabu = False
        
        # Generate all possible insertion moves
        # Move job from i to j (i != j)
        for i in range(n_jobs):
            job = current_seq[i]
            for j in range(n_jobs):
                if i == j:
                    continue
                
                # Check if this move is tabu
                # A move is tabu if the job was moved TO position j within the last tabu_tenure iterations
                # Note: When we move a job from i to j, the jobs in between shift.
                # So the "to_pos" in the tabu list should correspond to the position in the NEW sequence.
                # Let's define the move as (job, target_index_in_new_seq).
                # If we remove job at i and insert at j:
                # If j > i, the job is inserted at index j in the new sequence?
                # Let's trace: [A, B, C, D], move A (i=0) to j=2 (after C).
                # New: [B, C, A, D]. A is at index 2.
                # So target index is j if j > i, and j if j < i?
                # If j < i: [A, B, C, D], move D (i=3) to j=1 (after B).
                # New: [A, D, B, C]. D is at index 1.
                # So the target index in the new sequence is always j, EXCEPT if we consider the shift.
                # Actually, the standard definition is: remove i, insert at j.
                # The new position of the job is j if j <= i, and j-1 if j > i?
                # Let's check: [A, B, C, D], move A (i=0) to j=2.
                # Remove A: [B, C, D]. Insert at 2: [B, C, A, D]. A is at index 2.
                # So if j > i, the new index is j-1? No, it's j.
                # Wait. [B, C, D]. Insert at index 2 means before D. So [B, C, A, D]. Index 2.
                # So the new index is j.
                
                # Let's just store (job, j) in the tabu list.
                # Check if (job, j) is in the tabu list and not expired.
                is_tabu = False
                for (t_job, t_pos, exp) in tabu_list:
                    if t_job == job and t_pos == j and iteration - exp < tabu_tenure:
                        is_tabu = True
                        break
                
                # Calculate makespan for this move
                new_seq = list(current_seq)
                job_to_move = new_seq.pop(i)
                if j > i:
                    insert_idx = j - 1
                else:
                    insert_idx = j
                new_seq.insert(insert_idx, job_to_move)
                
                new_makespan = _calc_makespan(processing_times, new_seq)
                
                # Aspiration criterion: accept tabu move if it improves the best solution found so far
                if new_makespan < best_makespan:
                    is_tabu = False
                
                if new_makespan < best_move_makespan:
                    best_move_makespan = new_makespan
                    best_move = (i, j)
                    best_move_is_tabu = is_tabu
        
        # Clean up expired tabu entries
        tabu_list = [(j, p, e) for (j, p, e) in tabu_list if iteration - e >= tabu_tenure]
        
        if best_move is not None:
            i, j = best_move
            
            # Perform the move
            new_seq = list(current_seq)
            job_to_move = new_seq.pop(i)
            if j > i:
                insert_idx = j - 1
            else:
                insert_idx = j
            new_seq.insert(insert_idx, job_to_move)
            
            # Add to tabu list
            # The job was moved to position j in the new sequence?
            # Let's store (job_to_move, insert_idx, iteration)
            tabu_list.append((job_to_move, insert_idx, iteration))
            
            current_seq = new_seq
            current_makespan = best_move_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = list(current_seq)
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Diversification: if stuck for too long, restart with a random perturbation
            if no_improve_count > max_no_improve:
                # Perturb the current solution
                if n_jobs >= 2:
                    i1, i2 = random.sample(range(n_jobs), 2)
                    current_seq[i1], current_seq[i2] = current_seq[i2], current_seq[i1]
                    current_makespan = _calc_makespan(processing_times, current_seq)
                    no_improve_count = 0
                    # Clear tabu list to allow exploration
                    tabu_list = []
                    
        else:
            # No valid move found (should not happen)
            break
            
    return best_seq, best_makespan


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Generate diverse initial candidates
    candidates = []
    
    # V-shaped heuristic
    candidates.append(_v_shaped_heuristic(processing_times))
    
    # NEH heuristic
    candidates.append(_neh_heuristic(processing_times))
    
    # Random permutations
    for _ in range(3):
        candidates.append(_random_permutation(n_jobs))
    
    # Evaluate all candidates and keep the best
    best_seq = None
    best_makespan = float('inf')
    
    for cand in candidates:
        if time.time() >= deadline:
            break
        ms = _calc_makespan(processing_times, cand)
        if ms < best_makespan:
            best_makespan = ms
            best_seq = list(cand)
    
    if best_seq is None:
        best_seq = list(range(n_jobs))
        best_makespan = _calc_makespan(processing_times, best_seq)
    
    # Tabu Search
    if time.time() < deadline - 2.0:
        ts_deadline = deadline - 1.0
        ts_seq, ts_makespan = _tabu_search(processing_times, best_seq, ts_deadline)
        if ts_makespan < best_makespan:
            best_seq = ts_seq
            best_makespan = ts_makespan
    
    return best_seq


# EVOLVE-BLOCK-END
