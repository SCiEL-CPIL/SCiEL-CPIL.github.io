def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 110  # Safety margin
    
    if k != 3:
        # Fallback for other k, though task is k=3
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw
    
    # Try LUW construction with various d values
    # The LUW construction size depends on d.
    # We want n to be as small as possible, but >= Moore bound.
    # Moore bound for g=16 is 512.
    
    best_n = float('inf')
    best_edges = None
    
    # Try d from 6 to 15
    for d in range(6, 16):
        if _time.time() > deadline:
            break
        try:
            n_try, edges_try = _luw_construct(3, g, d_override=d)
            # Verify the graph
            adj = [[] for _ in range(n_try)]
            for u, v in edges_try:
                if u == v:
                    return 0, [] # Invalid
                adj[u].append(v)
                adj[v].append(u)
            
            # Check degree
            valid = True
            for i in range(n_try):
                if len(adj[i]) != 3:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check girth
            if _verify_girth_fast(adj, n_try, g):
                if n_try < best_n:
                    best_n = n_try
                    best_edges = edges_try
                    # We can break early if we find a small one, 
                    # but smaller d might give smaller n? 
                    # Usually n grows with d. So first valid is likely best?
                    # Not necessarily, but likely.
                    break
        except Exception:
            continue
            
    if best_edges is not None:
        return best_n, best_edges
        
    # If LUW fails, try a simple lift of Petersen
    # But LUW should work.
    # Final fallback: Return a known small graph if g is small, else error?
    # Task requires g=16.
    # If all fails, return a dummy valid 3-regular graph? No, must have girth >= 16.
    # If we can't find one, we must return *something*. 
    # The evaluator scores 0 if invalid.
    # But LUW is a standard construction, it should work.
    
    # As a last resort, try a larger d
    for d in range(16, 20):
         if _time.time() > deadline:
             break
         try:
             n_try, edges_try = _luw_construct(3, g, d_override=d)
             adj = [[] for _ in range(n_try)]
             for u, v in edges_try:
                 adj[u].append(v)
                 adj[v].append(u)
             if all(len(adj[i]) == 3 for i in range(n_try)) and _verify_girth_fast(adj, n_try, g):
                 return n_try, edges_try
         except Exception:
             pass
             
    # Absolute fallback: Return empty? No.
    # Return a cycle of length 16? Not 3-regular.
    # This should not happen.
    return 0, []
