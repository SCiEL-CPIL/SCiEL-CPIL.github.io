<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = P['n']
    m = P['m']
    
    # Initial family
    current_family = _greedy_chain_init(n, m)
    
    # 2-opt local search
    best_family, best_score = _two_opt_local_search(current_family, n, m, deadline)
    
    # Convert back to lists of ints
    result = []
    for s in best_family:
        lst = []
        for b in range(n):
            if s & (1 << b):
                lst.append(b)
        result.append(lst)
    
    return result
=======
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = P['n']
    m = P['m']
    
    # Run multiple restarts to escape local optima
    best_family = None
    best_score = -1
    
    # First, run greedy init + 2-opt
    current_family = _greedy_chain_init(n, m)
    fam, score = _two_opt_local_search(current_family, n, m, deadline)
    best_family = fam
    best_score = score
    
    # Check if we have time for more restarts
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    
    if remaining > 10.0:
        # Try a few more restarts with different random seeds
        for restart in range(5):
            if time.time() >= deadline:
                break
            # Perturb the best solution
            perturbed = list(best_family)
            for _ in range(5):
                idx = random.randint(0, m - 1)
                new_set = random.randint(0, (1 << n) - 1)
                if new_set not in perturbed:
                    perturbed[idx] = new_set
            
            fam, score = _two_opt_local_search(perturbed, n, m, deadline)
            if score > best_score:
                best_score = score
                best_family = fam
    
    # Final polish: one more round of 2-opt on the best solution
    if time.time() < deadline - 2.0:
        fam, score = _two_opt_local_search(best_family, n, m, deadline)
        if score > best_score:
            best_score = score
            best_family = fam
    
    # Convert back to lists of ints
    result = []
    for s in best_family:
        lst = []
        for b in range(n):
            if s & (1 << b):
                lst.append(b)
        result.append(lst)
    
    return result
>>>>>>> REPLACE
