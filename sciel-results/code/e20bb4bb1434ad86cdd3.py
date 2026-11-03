# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_n = float('inf')
    best_edges = None
    
    # 1. Try random lifts of Petersen graph (extended range)
    if time.time() < deadline - 30:
        petersen_edges = _get_petersen_edges()
        # Try M up to 100 to potentially find graphs near 936
        for M in range(52, 101):
            if time.time() > deadline - 30:
                break
            n = 10 * M
            if n >= best_n:
                continue
            # Increase trials for larger M if needed, but keep total time bounded
            trials = 10 if M < 80 else 20
            for _ in range(trials):
                if time.time() > deadline - 30:
                    break
                edges = _random_lift(petersen_edges, M)
                if edges is not None and _has_girth_at_least(n, edges, g):
                    best_n = n
                    best_edges = edges
                    break
            if best_n == n:
                # Found a valid graph for this M, can we do better?
                # Since n increases with M, if we found one for M, 
                # smaller M would have been checked first. 
                # But we started at 52. If 52-60 failed, and 61 succeeded, 
                # we have n=610. 
                # We should continue to see if a smaller n was missed? 
                # No, we iterate M increasing. 
                # If we found a valid graph at M, it's the smallest M that worked 
                # in this sequence? 
                # Not necessarily, because random failure. 
                # But if we find one at M=61 (n=610), it's better than LUW (likely >900).
                # We can break if we are confident, but let's try to find even smaller?
                # We started at 52. If 52-60 failed, 61 is the best so far.
                # We can break to save time for LUW if needed, or continue to see if 62 works? 
                # No, 62 is larger. 
                # So if we find a valid graph, we can break out of M loop?
                # Yes, because we want min n.
                break
        
        if best_n != float('inf'):
            return best_n, best_edges

    # 2. Deterministic LUW Construction
    min_d = max(4, g // 2 - 1)
    max_d = g + 5
    
    # Scan all d to find the one with minimum n
    candidates = []
    for d in range(min_d, max_d + 1):
        if time.time() > deadline - 5:
            break
            
        try:
            n_luw, edges_luw = _construct_luw_optimized(k, d)
        except Exception:
            continue
            
        if n_luw < best_n:
            if _has_girth_at_least(n_luw, edges_luw, g):
                candidates.append((n_luw, edges_luw))
                best_n = n_luw
                best_edges = edges_luw

    if best_n == float('inf'):
        # Fallback to largest d if no valid graph found in range
        n_luw, edges_luw = _construct_luw_optimized(k, max_d)
        return n_luw, edges_luw

    return best_n, best_edges

# ... (rest of the functions remain the same)
