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

def _tabu_search(processing_times, order, time_limit=10.0):
    """
    Tabu search with swap and insertion moves.
    Uses a tabu list to prohibit recent moves and enforce diversification.
    Aspiration criterion: a tabu move is allowed if it improves the global best.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan(processing_times, order)
    
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    global_best_order = list(current_order)
    global_best_ms = current_ms
    
    start_time = time.time()
    
    # Tabu list: store (position_i, position_j, job_at_i, job_at_j) or similar
    # For swaps: (i, j)
    # For insertions: (job, from_pos, to_pos)
    # We'll use a simple list of tuples and a tabu tenure
    tabu_tenure = max(5, n_jobs // 2)
    tabu_list = []  # list of (move_type, data, time_index)
    
    time_index = 0
    
    while time.time() - start_time < time_limit:
        best_move = None
        best_move_ms = current_ms
        best_move_type = None
        best_move_data = None
        
        # Explore neighborhood: swaps and insertions
        # To keep it fast, we limit the neighborhood size if n_jobs is large
        # But for flow shop, n_jobs is usually small (10-50)
        
        # 1. Swap moves
        for i in range(n_jobs):
            for j in range(i + 1, n_jobs):
                # Check if move is tabu
                is_tabu = False
                for (t, data, ti) in tabu_list:
                    if t == 'swap' and data == (i, j) and time_index - ti < tabu_tenure:
                        is_tabu = True
                        break
                    elif t == 'swap' and data == (j, i) and time_index - ti < tabu_tenure:
                        is_tabu = True
                        break
                
                # Perform swap
                current_order[i], current_order[j] = current_order[j], current_order[i]
                new_ms = _makespan(processing_times, current_order)
                
                # Aspiration criterion
                if is_tabu and new_ms >= global_best_ms:
                    # Revert
                    current_order[i], current_order[j] = current_order[j], current_order[i]
                    continue
                
                if new_ms < best_move_ms:
                    best_move_ms = new_ms
                    best_move_type = 'swap'
                    best_move_data = (i, j)
                
                # Revert
                current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # 2. Insertion moves (move job from pos i to pos j)
        for i in range(n_jobs):
            for j in range(n_jobs):
                if i == j or i == j + 1:
                    continue
                
                # Check if move is tabu
                is_tabu = False
                for (t, data, ti) in tabu_list:
                    if t == 'insert' and data == (i, j) and time_index - ti < tabu_tenure:
                        is_tabu = True
                        break
                
                # Perform insertion
                job = current_order.pop(i)
                # Adjust j if i < j
                insert_pos = j if i < j else j
                current_order.insert(insert_pos, job)
                
                new_ms = _makespan(processing_times, current_order)
                
                # Aspiration criterion
                if is_tabu and new_ms >= global_best_ms:
                    # Revert
                    current_order.pop(insert_pos)
                    current_order.insert(i, job)
                    continue
                
                if new_ms < best_move_ms:
                    best_move_ms = new_ms
                    best_move_type = 'insert'
                    best_move_data = (i, j)
                
                # Revert
                current_order.pop(insert_pos)
                current_order.insert(i, job)
        
        if best_move_type is None:
            # No improving or non-tabu moves found? Should not happen with tabu.
            # If best_move_ms == current_ms and no move accepted, we might be stuck.
            # Actually, we always accept the best non-tabu move, or best tabu if aspiration.
            # If no moves were evaluated (impossible), break.
            break
        
        # Apply best move
        if best_move_type == 'swap':
            i, j = best_move_data
            current_order[i], current_order[j] = current_order[j], current_order[i]
            tabu_list.append(('swap', (i, j), time_index))
        elif best_move_type == 'insert':
            i, j = best_move_data
            job = current_order.pop(i)
            insert_pos = j if i < j else j
            current_order.insert(insert_pos, job)
            tabu_list.append(('insert', (i, j), time_index))
        
        current_ms = best_move_ms
        
        if current_ms < global_best_ms:
            global_best_ms = current_ms
            global_best_order = list(current_order)
        
        time_index += 1
        
        # Clean up old tabu entries
        tabu_list = [(t, d, ti) for (t, d, ti) in tabu_list if time_index - ti < tabu_tenure]
        
        # Periodic diversification: if no improvement for a while, perturb
        # Simple check: if time_index % 100 == 0 and current_ms == global_best_ms
        # We'll skip complex diversification for now to keep it simple and fast.
        
        if time.time() - start_time > time_limit:
            break
    
    return global_best_order, global_best_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    
    Uses NEH heuristic followed by time-limited tabu search with restarts.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Generate initial solution using NEH
    initial_order = _neh_rule(processing_times)
    
    # Run tabu search with time budget
    remaining_time = deadline - time.time()
    if remaining_time < 5.0:
        return initial_order
    
    # Allocate most time to the first tabu search
    time_limit = remaining_time - 5.0  # Keep 5s safety
    best_order, best_ms = _tabu_search(
        processing_times, initial_order, 
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
            
            # Quick tabu search with shorter time limit
            tl = min(remaining_time - 5.0, 10.0)
            if tl <= 0:
                break
            order, ms = _tabu_search(
                processing_times, random_order,
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
