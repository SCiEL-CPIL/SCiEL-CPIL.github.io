<<<<<<< SEARCH
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        res = _try_rank(n, m, p, target_rank, T, num_trials=3, max_iter=80, deadline=deadline)
        if res is not None:
            result = res
            best_rank = target_rank
            # Don't break - try to find even lower rank
            # But if we found a good one, we can stop trying much higher ranks
    
    return result
=======
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Reduce trials and iterations to ensure we finish in time
        # If we find a solution, we update best_rank
        res = _try_rank(n, m, p, target_rank, T, num_trials=2, max_iter=50, deadline=deadline)
        if res is not None:
            result = res
            best_rank = target_rank
    
    return result
>>>>>>> REPLACE
