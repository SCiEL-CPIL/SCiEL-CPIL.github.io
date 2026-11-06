"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math


def _calc_makespan(processing_times, perm):
    """
    Calculate makespan for a given permutation using the flow-shop recurrence.
    """
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    
    n_machines = processing_times.shape[1]
    if n_machines == 1:
        return float(processing_times[perm, 0].sum())
    
    pt = processing_times[perm]
    
    # Completion time matrix: C[i][j] = completion time of job i on machine j
    # We only need the last row's last element, but we track the previous row.
    # C[i][j] = max(C[i-1][j], C[i][j-1]) + p[i][j]
    
    # Initialize with the first job
    prev_row = np.empty(n_machines, dtype=float)
    prev_row[0] = pt[0, 0]
    for j in range(1, n_machines):
        prev_row[j] = prev_row[j-1] + pt[0, j]
    
    if n_jobs == 1:
        return float(prev_row[-1])
    
    curr_row = np.empty(n_machines, dtype=float)
    
    for i in range(1, n_jobs):
        curr_row[0] = prev_row[0] + pt[i, 0]
        for j in range(1, n_machines):
            curr_row[j] = (prev_row[j] if prev_row[j] > curr_row[j-1] else curr_row[j-1]) + pt[i, j]
        prev_row, curr_row = curr_row, prev_row
    
    return float(prev_row[-1])


def _calc_makespan_fast(processing_times, perm, prev_row=None):
    """
    Calculate makespan, optionally reusing a previous row for speed (incremental).
    Returns (makespan, last_row) for potential reuse.
    """
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0, None
    
    n_machines = processing_times.shape[1]
    if n_machines == 1:
        return float(processing_times[perm, 0].sum()), None
    
    pt = processing_times[perm]
    
    if prev_row is not None and len(prev_row) == n_machines:
        # This is tricky for full recalculation; just do standard.
        # For now, standard calculation.
        pass
    
    prev_row = np.empty(n_machines, dtype=float)
    prev_row[0] = pt[0, 0]
    for j in range(1, n_machines):
        prev_row[j] = prev_row[j-1] + pt[0, j]
    
    if n_jobs == 1:
        return float(prev_row[-1]), prev_row
    
    curr_row = np.empty(n_machines, dtype=float)
    
    for i in range(1, n_jobs):
        curr_row[0] = prev_row[0] + pt[i, 0]
        for j in range(1, n_machines):
            curr_row[j] = (prev_row[j] if prev_row[j] > curr_row[j-1] else curr_row[j-1]) + pt[i, j]
        prev_row, curr_row = curr_row, prev_row
    
    return float(prev_row[-1]), prev_row


def _insertion_move(sequence, i, j):
    """
    Remove element at position i and insert it at position j.
    If j > i, the effective insertion index is j-1 because the element at i
    was removed, shifting everything after i to the left by 1.
    """
    new_seq = list(sequence)
    job = new_seq.pop(i)
    if j > i:
        insert_idx = j - 1
    else:
        insert_idx = j
    new_seq.insert(insert_idx, job)
    return new_seq


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


def _v_shape_heuristic(processing_times):
    """
    V-shaped heuristic:
    1. Sort jobs by total processing time.
    2. Build sequence by adding jobs to the left or right end of the current sequence
       to minimize makespan (or a proxy).
    """
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = np.argsort(totals)  # Ascending for V-shape (shortest at ends? No, usually longest at ends for V)
    # Actually, V-shape usually means: sequence is non-decreasing then non-increasing.
    # Or: start with the job with largest total, add others to left or right.
    
    # Let's try: Start with the job having the largest total.
    # Then for each remaining job (sorted by total descending), add to left or right.
    
    order_desc = np.argsort(-totals)
    
    # Initialize with the first job
    sequence = [order_desc[0]]
    
    # We need to evaluate makespan for adding to left vs right.
    # To be efficient, we can just compute full makespan for small n, or use a proxy.
    # For now, compute full makespan for each candidate position (left, right).
    
    for i in range(1, n_jobs):
        job = order_desc[i]
        
        # Option 1: Add to left
        seq_left = [job] + sequence
        ms_left = _calc_makespan(processing_times, seq_left)
        
        # Option 2: Add to right
        seq_right = sequence + [job]
        ms_right = _calc_makespan(processing_times, seq_right)
        
        if ms_left < ms_right:
            sequence = seq_left
        else:
            sequence = seq_right
    
    return sequence


def _tabu_search(processing_times, initial_seq, deadline, tabu_tenure=7, max_iterations=None):
    """
    Tabu search with insertion moves.
    """
    n_jobs = len(initial_seq)
    if n_jobs < 2:
        return list(initial_seq), _calc_makespan(processing_times, initial_seq)
    
    current_seq = list(initial_seq)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    # Tabu list: store (job, position) or (i, j) moves
    # We'll store the (i, j) pair of the last `tabu_tenure` moves.
    tabu_list = []
    
    iteration = 0
    if max_iterations is None:
        max_iterations = 100000  # Safety cap
    
    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        best_move_makespan = float('inf')
        best_move_i = -1
        best_move_j = -1
        best_move_seq = None
        
        # Check all insertion moves
        for i in range(n_jobs):
            for j in range(n_jobs):
                if i == j:
                    continue
                
                # Check tabu status
                # A move is tabu if it reverses a recent move.
                # For insertion, the "move" is defined by (job, from_pos, to_pos).
                # Simpler: store the resulting permutation's hash? No, too expensive.
                # Let's store (i, j) as the move key.
                move_key = (i, j)
                if move_key in tabu_list:
                    continue
                
                new_seq = _insertion_move(current_seq, i, j)
                new_makespan = _calc_makespan(processing_times, new_seq)
                
                # Aspiration criterion: if it improves the best, allow it
                if new_makespan < best_makespan or new_makespan < best_move_makespan:
                    best_move_makespan = new_makespan
                    best_move_i = i
                    best_move_j = j
                    best_move_seq = new_seq
        
        if best_move_seq is not None:
            # Add current move to tabu list
            tabu_list.append((best_move_i, best_move_j))
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
            
            current_seq = best_move_seq
            current_makespan = best_move_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = list(current_seq)
        else:
            # No non-tabu improving move found. Try to break tabu or restart?
            # For now, break to avoid infinite loop if stuck.
            break
    
    return best_seq, best_makespan


def _simulated_annealing(processing_times, sequence, deadline):
    """
    Simulated annealing with insertion perturbations.
    """
    n_jobs = len(sequence)
    if n_jobs < 2:
        return list(sequence), _calc_makespan(processing_times, sequence)
    
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, sequence)
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    T = current_makespan * 0.1
    cooling_rate = 0.9995
    min_T = 1e-6
    
    while time.time() < deadline and T > min_T:
        # Use insertion move for perturbation
        i, j = random.sample(range(n_jobs), 2)
        new_seq = _insertion_move(current_seq, i, j)
        new_makespan = _calc_makespan(processing_times, new_seq)
        
        delta = new_makespan - current_makespan
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_seq = new_seq
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = list(current_seq)
        
        T *= cooling_rate
    
    return best_seq, best_makespan


def _local_search(processing_times, sequence, deadline):
    """
    First-improvement local search with insertion moves.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        for i in range(n_jobs):
            if time.time() >= deadline:
                return current_seq, current_makespan
            for j in range(n_jobs):
                if j == i:
                    continue
                if time.time() >= deadline:
                    return current_seq, current_makespan
                
                new_seq = _insertion_move(current_seq, i, j)
                new_makespan = _calc_makespan(processing_times, new_seq)
                
                if new_makespan < current_makespan:
                    current_seq = new_seq
                    current_makespan = new_makespan
                    improved = True
                    break  # restart scan from i=0
            if improved:
                break
    
    return current_seq, current_makespan


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
    
    # 1. NEH
    candidates.append(_neh_heuristic(processing_times))
    
    # 2. V-shape
    candidates.append(_v_shape_heuristic(processing_times))
    
    # 3. Random
    candidates.append(list(np.random.permutation(n_jobs)))
    
    # 4. Reversed NEH
    candidates.append(list(reversed(_neh_heuristic(processing_times))))
    
    # Evaluate and find the best initial candidate
    best_seq = None
    best_makespan = float('inf')
    
    for cand in candidates:
        ms = _calc_makespan(processing_times, cand)
        if ms < best_makespan:
            best_makespan = ms
            best_seq = list(cand)
    
    if best_seq is None:
        best_seq = list(range(n_jobs))
        best_makespan = _calc_makespan(processing_times, best_seq)
    
    # Phase 1: Local search polish
    if time.time() < deadline - 60.0:
        ls_deadline = min(deadline - 60.0, time.time() + 20.0)
        ls_seq, ls_makespan = _local_search(processing_times, best_seq, ls_deadline)
        if ls_makespan < best_makespan:
            best_seq = ls_seq
            best_makespan = ls_makespan
    
    # Phase 2: Tabu Search (main workhorse)
    if time.time() < deadline - 20.0:
        ts_deadline = min(deadline - 20.0, time.time() + 50.0)
        ts_seq, ts_makespan = _tabu_search(processing_times, best_seq, ts_deadline, tabu_tenure=7)
        if ts_makespan < best_makespan:
            best_seq = ts_seq
            best_makespan = ts_makespan
    
    # Phase 3: Simulated Annealing (to escape local optima)
    if time.time() < deadline - 10.0:
        sa_deadline = min(deadline - 10.0, time.time() + 30.0)
        sa_seq, sa_makespan = _simulated_annealing(processing_times, best_seq, sa_deadline)
        if sa_makespan < best_makespan:
            best_seq = sa_seq
            best_makespan = sa_makespan
    
    # Phase 4: Final Local Search
    if time.time() < deadline - 1.0:
        final_seq, final_makespan = _local_search(processing_times, best_seq, deadline)
        if final_makespan < best_makespan:
            best_seq = final_seq
            best_makespan = final_makespan
    
    return best_seq


# EVOLVE-BLOCK-END
