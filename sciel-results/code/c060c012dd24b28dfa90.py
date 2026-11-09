<<<<<<< SEARCH
def _insertion(perm, processing_times):
    """Local search using insertion moves."""
    n = len(perm)
    m = processing_times.shape[1]
    current_ms = _makespan_fast(perm, processing_times)
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            job = perm[i]
            for j in range(n):
                if i == j:
                    continue
                # Move job from i to j
                new_perm = perm.copy()
                new_perm.pop(i)
                # Adjust j if needed
                if i < j:
                    j -= 1
                new_perm.insert(j, job)
                
                new_ms = _makespan_fast(new_perm, processing_times)
                if new_ms < current_ms:
                    perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break
            if improved:
                break
    
    return perm, current_ms
=======
def _insertion(perm, processing_times):
    """Local search using insertion moves."""
    n = len(perm)
    m = processing_times.shape[1]
    current_ms = _makespan_fast(perm, processing_times)
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            job = perm[i]
            for j in range(n):
                if i == j:
                    continue
                # Move job from i to j
                new_perm = perm.copy()
                new_perm.pop(i)
                # Adjust j if needed
                if i < j:
                    j -= 1
                new_perm.insert(j, job)
                
                new_ms = _makespan_fast(new_perm, processing_times)
                if new_ms < current_ms:
                    perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break
            if improved:
                break
    
    return perm, current_ms

def _tabu_search(perm, processing_times, deadline, best_ms_ref, best_perm_ref):
    """Tabu search with insertion moves and aspiration criteria."""
    n = len(perm)
    m = processing_times.shape[1]
    pt = processing_times
    
    current_perm = list(perm)
    current_ms = _makespan_fast(current_perm, pt)
    
    # Tabu list: store (job, from_pos, to_pos) with expiration
    tabu_list = []
    tabu_tenure = max(5, n // 2)
    
    # Track best found
    local_best_ms = current_ms
    local_best_perm = list(current_perm)
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline - 2.0:
        if no_improve > max_no_improve:
            break
        
        # Generate all possible insertion moves
        best_move = None
        best_move_ms = float('inf')
        best_move_aspiration = False
        best_move_job = -1
        best_move_from = -1
        best_move_to = -1
        
        for i in range(n):
            job = current_perm[i]
            for j in range(n):
                if i == j:
                    continue
                
                # Create new permutation with job moved from i to j
                new_perm = current_perm.copy()
                new_perm.pop(i)
                if i < j:
                    j -= 1
                new_perm.insert(j, job)
                
                new_ms = _makespan_fast(new_perm, pt)
                
                # Check if move is tabu
                is_tabu = False
                for move, exp in tabu_list:
                    if move[0] == job and move[1] == i and move[2] == j:
                        if exp > len(tabu_list) - len([x for x in tabu_list if x[1] > len(tabu_list)]) :
                            # Simplified: just check if in tabu list
                            is_tabu = True
                            break
                
                # More efficient tabu check
                is_tabu = any(mv[0] == job and mv[1] == i and mv[2] == j for mv, _ in tabu_list)
                
                # Aspiration: accept tabu move if it improves global best
                if is_tabu and new_ms >= best_ms_ref:
                    continue
                
                if new_ms < best_move_ms:
                    best_move_ms = new_ms
                    best_move_job = job
                    best_move_from = i
                    best_move_to = j
                    best_move_aspiration = is_tabu
        
        if best_move_job == -1:
            break
        
        # Apply best move
        job = best_move_job
        from_pos = best_move_from
        to_pos = best_move_to
        
        new_perm = current_perm.copy()
        new_perm.pop(from_pos)
        if from_pos < to_pos:
            to_pos -= 1
        new_perm.insert(to_pos, job)
        
        current_perm = new_perm
        current_ms = best_move_ms
        
        # Update tabu list
        tabu_list.append(((job, from_pos, to_pos), tabu_tenure))
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        
        # Update local best
        if current_ms < local_best_ms:
            local_best_ms = current_ms
            local_best_perm = list(current_perm)
            no_improve = 0
        else:
            no_improve += 1
        
        # Update global best
        if current_ms < best_ms_ref[0]:
            best_ms_ref[0] = current_ms
            best_perm_ref[0] = list(current_perm)
    
    return local_best_perm, local_best_ms
>>>>>>> REPLACE

