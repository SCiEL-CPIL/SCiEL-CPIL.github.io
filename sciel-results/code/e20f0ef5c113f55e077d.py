<<<<<<< SEARCH
def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    # 1. Generate initial solution using NEH heuristic
    current_order = _neh_heuristic(processing_times)
    current_makespan = _makespan(current_order, processing_times)
    
    best_order = current_order[:]
    best_makespan = current_makespan
    
    # 2. Local Search
    # Allocate 70% of time to local search, 30% to SA
    # Or just run LS until no improvement, then SA with remaining time.
    # LS is deterministic and fast.
    ls_deadline = start_time + (deadline - start_time) * 0.7
    
    current_order, current_makespan = _local_search(current_order, processing_times, ls_deadline)
    
    if current_makespan < best_makespan:
        best_makespan = current_makespan
        best_order = current_order[:]
        
    # 3. Simulated Annealing
    if time.time() < deadline:
        # Use a higher initial temperature for SA to explore more
        # Scale temp based on typical makespan magnitude? 
        # 1000 is a reasonable default for synthetic data.
        sa_order, sa_ms = _simulated_annealing(current_order, processing_times, deadline, initial_temp=500.0, cooling=0.9995)
        
        if sa_ms < best_makespan:
            best_makespan = sa_ms
            best_order = sa_order[:]
            
    return best_order
=======
def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    # 1. Generate initial solution using NEH heuristic
    current_order = _neh_heuristic(processing_times)
    current_makespan = _makespan(current_order, processing_times)
    
    best_order = current_order[:]
    best_makespan = current_makespan
    
    # 2. Local Search
    # Allocate 50% of time to local search, 50% to SA
    ls_deadline = start_time + (deadline - start_time) * 0.5
    
    current_order, current_makespan = _local_search(current_order, processing_times, ls_deadline)
    
    if current_makespan < best_makespan:
        best_makespan = current_makespan
        best_order = current_order[:]
        
    # 3. Simulated Annealing with multiple restarts
    if time.time() < deadline:
        remaining_time = deadline - time.time()
        
        # Run SA with multiple restarts if time permits
        n_restarts = max(1, int(remaining_time / 5.0))  # ~5s per restart
        n_restarts = min(n_restarts, 10)
        
        for restart in range(n_restarts):
            if time.time() >= deadline:
                break
            
            restart_deadline = deadline - 0.5  # safety margin
            
            # For restarts > 0, perturb the best solution
            if restart == 0:
                sa_start = current_order[:]
            else:
                # Create a perturbed version of best solution
                sa_start = best_order[:]
                # Randomly swap a few pairs
                n_swaps = max(1, n_jobs // 5)
                for _ in range(n_swaps):
                    i = random.randint(0, n_jobs - 1)
                    j = random.randint(0, n_jobs - 1)
                    if i != j:
                        sa_start[i], sa_start[j] = sa_start[j], sa_start[i]
            
            sa_order, sa_ms = _simulated_annealing(sa_start, processing_times, restart_deadline, 
                                                   initial_temp=300.0, cooling=0.998)
            
            if sa_ms < best_makespan:
                best_makespan = sa_ms
                best_order = sa_order[:]
                current_order = sa_order[:]
                current_makespan = sa_ms
            
            # Quick local search polish on SA result
            if time.time() < deadline - 2.0 and sa_ms < best_makespan:
                ls_polish_deadline = min(deadline - 1.0, time.time() + 2.0)
                polished_order, polished_ms = _local_search(sa_order, processing_times, ls_polish_deadline)
                if polished_ms < best_makespan:
                    best_makespan = polished_ms
                    best_order = polished_order[:]
                    current_order = polished_order[:]
                    current_makespan = polished_ms
    
    return best_order
>>>>>>> REPLACE
