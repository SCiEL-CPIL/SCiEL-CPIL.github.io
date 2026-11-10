import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 12
    MAX_L = 144
    M = 27720
    
    best_m = 0
    best_L = 0
    
    # Pre-allocate arrays for matching
    match_right = [-1] * (MAX_L + 1)
    visited = [False] * (MAX_L + 1)
    
    for r in range(M):
        if time.time() > deadline:
            break
            
        # Compute valid offsets for this r
        valid_offsets = [[] for _ in range(N + 1)]
        for i in range(1, N + 1):
            rem = (-r) % i
            if rem == 0:
                for d in range(i, MAX_L + 1, i):
                    valid_offsets[i].append(d)
            else:
                for d in range(rem, MAX_L + 1, i):
                    if d > 0:
                        valid_offsets[i].append(d)
        
        # Define DFS using nonlocal L
        def try_kuhn(i, L):
            for d in valid_offsets[i]:
                if d > L:
                    break
                if match_right[d] == -1:
                    match_right[d] = i
                    return True
                if not visited[d]:
                    visited[d] = True
                    if try_kuhn(match_right[d], L):
                        match_right[d] = i
                        return True
            return False
        
        found_L = None
        for L in range(1, MAX_L + 1):
            # Reset match_right? 
            # No, Kuhn's algorithm builds the matching incrementally.
            # But for a new L, the graph changes (more edges).
            # We can reuse the previous matching?
            # If a matching exists for L, it exists for L+1.
            # We want the MINIMAL L.
            # So we start from scratch for each L?
            # Or we can start with the matching from L-1 and try to augment?
            # Since we are looking for the first L where a perfect matching exists,
            # and we check L=1, 2, ..., we can just re-run the matching for each L.
            # The cost is low.
            
            # However, we must reset match_right and visited for each L check?
            # Yes, because the graph is different.
            
            # Optimization: We can keep the match_right from the previous L?
            # If we have a matching for L-1, it is a valid partial matching for L.
            # We can try to extend it.
            # But Kuhn's algorithm as written finds a maximum matching from scratch.
            # Let's just re-initialize match_right for simplicity and correctness.
            # 144 * 27720 = 4 million re-initializations of a 145-list.
            # 4M * 145 ops = 580M ops. Might be slow.
            
            # Better: Only reset match_right if we failed?
            # No, for each L, we need to check if a perfect matching exists.
            
            # Let's try to be smarter.
            # We know that if a perfect matching exists for L, it exists for L+1.
            # We are looking for the FIRST such L.
            # So we must check L=1, then 2, etc.
            
            # To speed up, we can use the fact that the matching for L-1 is a subset of the matching for L?
            # Not necessarily, but we can use the previous matching as a starting point.
            # Initialize match_right with the result from L-1.
            
            # For L=1, match_right is all -1.
            # For L>1, we can start with the match_right from L-1.
            # We need to be careful: the indices d in match_right might be > L-1? No, they were <= L-1.
            # So they are valid for L.
            
            # Let's implement this incremental approach.
            
            if L == 1:
                for d in range(MAX_L + 1):
                    match_right[d] = -1
            
            # Run Kuhn's algorithm
            # We need to reset visited for each i
            matched = 0
            for i in range(1, N + 1):
                # Reset visited
                for d in range(MAX_L + 1):
                    visited[d] = False
                if try_kuhn(i, L):
                    matched += 1
                else:
                    break
            
            if matched == N:
                found_L = L
                break
        
        if found_L is not None:
            if found_L > best_L:
                best_L = found_L
                best_m = r
                
    return best_m
