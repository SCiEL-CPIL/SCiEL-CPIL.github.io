<<<<<<< SEARCH
    best_code = None
    best_size = float('inf')
    
    # Run multiple restarts
    for i in range(30):
        if time.time() > deadline - 10:
            break
            
        rng = np.random.default_rng(42 + i)
        
        # Greedy Init
        code = greedy_init(rng)
        
        # Local Search
        code = local_search(code, rng, deadline)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        if best_size <= 486:
            break
        
        # If we have time, try a second pass of local search on the best code found so far
        if time.time() < deadline - 15 and i > 0:
            code2 = local_search(best_code.copy(), rng, deadline)
            if len(code2) < best_size:
                best_size = len(code2)
                best_code = code2
=======
    best_code = None
    best_size = float('inf')
    
    # Run multiple restarts
    for i in range(40):
        if time.time() > deadline - 10:
            break
            
        rng = np.random.default_rng(42 + i)
        
        # Greedy Init
        code = greedy_init(rng)
        
        # Local Search
        code = local_search(code, rng, deadline)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        if best_size <= 486:
            break
        
        # If we have time, try a second pass of local search on the best code found so far
        if time.time() < deadline - 15 and i > 0:
            code2 = local_search(set(best_code), rng, deadline)
            if len(code2) < best_size:
                best_size = len(code2)
                best_code = code2
>>>>>>> REPLACE

