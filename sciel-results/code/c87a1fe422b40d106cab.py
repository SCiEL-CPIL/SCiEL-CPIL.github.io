<<<<<<< SEARCH
def _tabu_search(a, b, p, time_limit=None, start_time=None, tabu_len=20):
    """
    Tabu search with steepest descent using vectorized delta matrix.
    Allows temporarily worsening moves to escape local minima.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
        
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    # Tabu list: stores (i, j) pairs that are forbidden
    tabu = {}
    # Key: sorted tuple (min(i,j), max(i,j)) -> iteration when it becomes untabu
    iteration = 0
    best_cost = current_cost
    best_p = p.copy()
    
    while time.time() - start_time < time_limit:
        D = _compute_delta_matrix(a, b, p)
        
        # Find the best non-tabu move
        best_delta = float('inf')
        best_i, best_j = -1, -1
        best_is_tabu = True
        
        # Iterate over all pairs
        for i in range(n):
            for j in range(i + 1, n):
                delta = D[i, j]
                key = (i, j)
                is_tabu = key in tabu and tabu[key] > iteration
                
                if not is_tabu:
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
                        best_is_tabu = False
                else:
                    # Aspiration: allow tabu move if it improves best cost
                    if delta < best_cost:
                        if delta < best_delta:
                            best_delta = delta
                            best_i, best_j = i, j
                            best_is_tabu = False
        
        if best_i == -1:
            # No improving or non-tabu move found, try best tabu move
            for i in range(n):
                for j in range(i + 1, n):
                    delta = D[i, j]
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
        
        if best_i == -1:
            break
            
        # Perform the move
        i, j = best_i, best_j
        p[i], p[j] = p[j], p[i]
        current_cost += int(best_delta)
        
        # Update tabu list
        tabu[(i, j)] = iteration + tabu_len
        tabu[(j, i)] = iteration + tabu_len
        
        # Clean up expired tabu entries
        expired_keys = [k for k, v in tabu.items() if v <= iteration]
        for k in expired_keys:
            del tabu[k]
            
        iteration += 1
        
        # Track best solution
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    return best_p, best_cost
=======
def _tabu_search(a, b, p, time_limit=None, start_time=None, tabu_len=20):
    """
    Tabu search with steepest descent using vectorized delta matrix.
    Allows temporarily worsening moves to escape local minima.
    Optimized with vectorized tabu handling.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
        
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    # Tabu list: stores (i, j) pairs that are forbidden
    # Use a 2D array for faster lookup: tabu_matrix[i, j] = iteration when it becomes untabu
    tabu_matrix = np.zeros((n, n), dtype=np.int64)
    iteration = 0
    best_cost = current_cost
    best_p = p.copy()
    
    while time.time() - start_time < time_limit:
        D = _compute_delta_matrix(a, b, p)
        
        # Create a copy of D with tabu moves set to +infinity
        # But allow aspiration: tabu moves that improve best_cost are allowed
        D_tabu = D.copy()
        
        # Mark tabu moves
        tabu_mask = tabu_matrix > iteration
        # For tabu moves, set delta to inf unless aspiration applies (delta < best_cost)
        # We need to handle this carefully
        
        # First, set all tabu moves to inf
        D_tabu[tabu_mask] = np.inf
        
        # Now, for tabu moves that satisfy aspiration (delta < best_cost), restore their delta
        # But only if they are better than the current best non-tabu move
        # This is tricky to do vectorized, so let's do it in a simpler way:
        # Find the best non-tabu move first
        best_non_tabu_delta = np.min(D_tabu)
        best_non_tabu_idx = np.unravel_index(np.argmin(D_tabu), D_tabu.shape)
        
        # Now check if any tabu move with aspiration is better
        # Get all tabu moves that satisfy aspiration
        aspiration_mask = tabu_mask & (D < best_cost)
        if np.any(aspiration_mask):
            # Find the best aspiration move
            D_aspiration = D.copy()
            D_aspiration[~aspiration_mask] = np.inf
            best_aspiration_delta = np.min(D_aspiration)
            best_aspiration_idx = np.unravel_index(np.argmin(D_aspiration), D_aspiration.shape)
            
            if best_aspiration_delta < best_non_tabu_delta:
                best_delta = best_aspiration_delta
                best_i, best_j = best_aspiration_idx
            else:
                best_delta = best_non_tabu_delta
                best_i, best_j = best_non_tabu_idx
        else:
            best_delta = best_non_tabu_delta
            best_i, best_j = best_non_tabu_idx
        
        if best_delta >= np.inf:
            # No valid move found, break
            break
            
        # Perform the move
        i, j = best_i, best_j
        p[i], p[j] = p[j], p[i]
        current_cost += int(best_delta)
        
        # Update tabu list
        tabu_matrix[i, j] = iteration + tabu_len
        tabu_matrix[j, i] = iteration + tabu_len
        
        iteration += 1
        
        # Track best solution
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    return best_p, best_cost
>>>>>>> REPLACE
<<<<<<< SEARCH
def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    best_p = None
    best_cost = float('inf')
    
    # Phase 1: Try multiple initializations with local search
    initializations = []
    
    # Greedy initialization
    p = _greedy_initial(a, b)
    initializations.append(p)
    
    # Hungarian initialization
    p = _hungarian_initial(a, b)
    initializations.append(p)
    
    # Random initializations
    for _ in range(3):
        p = _random_initial(n)
        initializations.append(p)
    
    # Run local search on each initialization with time budget
    time_per_init = (total_deadline - 20) / len(initializations)
    
    for init_p in initializations:
        if time.time() - start_time > total_deadline - 15:
            break
        
        p = init_p.copy()
        current_cost = _qap_cost(a, b, p)
        
        # Local search to refine
        p, current_cost = _local_search_fast(a, b, p, time_limit=time_per_init, start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # Phase 2: Tabu search on the best solution found
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 10 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        
        p, current_cost = _tabu_search(a, b, p, time_limit=min(remaining_time - 5, 40), start_time=start_time, tabu_len=15)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # Phase 3: Final polish with SA on the best solution found
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 3 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        
        T_start = max(1.0, best_cost * 0.01)
        T_end = 0.1
        elapsed_total = total_deadline
        iters = 0
        
        while time.time() - start_time < total_deadline - 1:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = _compute_swap_delta(a, b, p, i, j)
            new_cost = current_cost + delta
            
            elapsed = time.time() - start_time
            T = T_start * (T_end / T_start) ** (elapsed / elapsed_total)
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            iters += 1
    
    if best_p is None:
        best_p = _greedy_initial(a, b)
    
    return best_p.tolist()
=======
def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    best_p = None
    best_cost = float('inf')
    
    # Phase 1: Try multiple initializations with local search
    initializations = []
    
    # Greedy initialization
    p = _greedy_initial(a, b)
    initializations.append(p)
    
    # Hungarian initialization
    p = _hungarian_initial(a, b)
    initializations.append(p)
    
    # Random initializations
    for _ in range(3):
        p = _random_initial(n)
        initializations.append(p)
    
    # Run local search on each initialization with time budget
    time_per_init = (total_deadline - 20) / len(initializations)
    
    for init_p in initializations:
        if time.time() - start_time > total_deadline - 15:
            break
        
        p = init_p.copy()
        current_cost = _qap_cost(a, b, p)
        
        # Local search to refine
        p, current_cost = _local_search_fast(a, b, p, time_limit=time_per_init, start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # Phase 2: Tabu search on the best solution found
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 10 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        
        p, current_cost = _tabu_search(a, b, p, time_limit=min(remaining_time - 5, 40), start_time=start_time, tabu_len=15)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # Phase 3: Final polish with SA on the best solution found
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 3 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        
        T_start = max(1.0, best_cost * 0.01)
        T_end = 0.1
        elapsed_total = total_deadline
        iters = 0
        
        while time.time() - start_time < total_deadline - 1:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = _compute_swap_delta(a, b, p, i, j)
            new_cost = current_cost + delta
            
            elapsed = time.time() - start_time
            T = T_start * (T_end / T_start) ** (elapsed / elapsed_total)
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            iters += 1
    
    if best_p is None:
        best_p = _greedy_initial(a, b)
    
    return best_p.tolist()
>>>>>>> REPLACE

BET: {mechanism: "Vectorized tabu search eliminates Python loop overhead for finding best non-tabu move, allowing more iterations within the same time budget", q: 0.65, delta: [0.001, 0.015]}
