<<<<<<< SEARCH
    # Start 1: Greedy initialization
    p = _greedy_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    
    # Local search to refine
    p, current_cost = _local_search(a, b, p, max_iter=50000, time_limit=total_deadline - 5, start_time=start_time)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # If we still have time, try additional random starts
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 10:
        # Try a few random starts
        num_random_starts = min(5, int(remaining_time // 10))
        for _ in range(num_random_starts):
            if time.time() - start_time > total_deadline - 5:
                break
            
            p = _random_initial(n)
            current_cost = _qap_cost(a, b, p)
            
            # Local search to refine
            p, current_cost = _local_search(a, b, p, max_iter=50000, time_limit=total_deadline - 5, start_time=start_time)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
=======
    # Start 1: Spectral initialization
    p = _spectral_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    
    # Local search to refine
    p, current_cost = _local_search(a, b, p, max_iter=50000, time_limit=total_deadline - 5, start_time=start_time)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # Start 2: Greedy initialization
    p = _greedy_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    
    # Local search to refine
    p, current_cost = _local_search(a, b, p, max_iter=50000, time_limit=total_deadline - 5, start_time=start_time)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # If we still have time, try additional random starts
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 10:
        # Try a few random starts
        num_random_starts = min(5, int(remaining_time // 10))
        for _ in range(num_random_starts):
            if time.time() - start_time > total_deadline - 5:
                break
            
            p = _random_initial(n)
            current_cost = _qap_cost(a, b, p)
            
            # Local search to refine
            p, current_cost = _local_search(a, b, p, max_iter=50000, time_limit=total_deadline - 5, start_time=start_time)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
>>>>>>> REPLACE

