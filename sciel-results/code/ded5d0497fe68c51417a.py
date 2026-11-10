<<<<<<< SEARCH
    # Additional local search restarts from best solution with perturbations
    for restart in range(50):
        if time.time() > deadline:
            break
        
        S = set(best_set)
        
        # Randomly remove some elements
        n_remove = rng.randint(1, min(5, len(S)))
        to_remove = rng.sample(list(S), n_remove)
        for x in to_remove:
            S.discard(x)
        
        # Add some random elements
        available = [x for x in range(1, _N + 1) if x not in S]
        if available:
            n_add = rng.randint(0, min(3, len(available)))
            to_add = rng.sample(available, n_add)
            for x in to_add:
                if x not in _get_blocked(S):
                    S.add(x)
        
        result = _local_search(S, deadline, rng)
        w = sum(_RECIP[x] for x in result)
        
        if w > best_weight:
            best_weight = w
            best_set = result.copy()
    
    return list(best_set)
=======
    # Additional local search restarts from best solution with perturbations
    for restart in range(50):
        if time.time() > deadline:
            break
        
        S = set(best_set)
        
        # Randomly remove some elements
        n_remove = rng.randint(1, min(5, len(S)))
        to_remove = rng.sample(list(S), n_remove)
        for x in to_remove:
            S.discard(x)
        
        # Add some random elements
        blocked_now, _ = _get_blocked_counts(S)
        available = [x for x in range(1, _N + 1) if x not in S and x not in blocked_now]
        if available:
            n_add = rng.randint(0, min(3, len(available)))
            to_add = rng.sample(available, n_add)
            for x in to_add:
                S.add(x)
        
        result = _local_search(S, deadline, rng)
        w = sum(_RECIP[x] for x in result)
        
        if w > best_weight:
            best_weight = w
            best_set = result.copy()
    
    # Final polish pass on the best solution
    if time.time() < deadline:
        polished = _polish(best_set, deadline)
        w = sum(_RECIP[x] for x in polished)
        if w > best_weight:
            best_weight = w
            best_set = polished
    
    return list(best_set)
>>>>>>> REPLACE
