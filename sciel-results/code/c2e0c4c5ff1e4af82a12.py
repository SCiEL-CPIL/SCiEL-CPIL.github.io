def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_edges = None
    best_count = float('inf')
    
    # Try n=10
    n = 10
    all_edges_k10 = [list(e) for e in itertools.combinations(range(n), 5)]
    
    # Multiple restarts of greedy removal
    for attempt in range(20): # Max 20 restarts
        if time.time() > deadline:
            break
            
        current_edges = all_edges_k10[:]
        random.shuffle(current_edges)
        
        removed_any = True
        # Local deadline for this attempt
        attempt_deadline = min(deadline, start_time + 5 + attempt * 2)
        
        while removed_any and (time.time() < attempt_deadline):
            removed_any = False
            random.shuffle(current_edges)
            
            for i in range(len(current_edges) - 1, -1, -1):
                if time.time() > attempt_deadline:
                    break
                e = current_edges[i]
                new_edges = current_edges[:i] + current_edges[i+1:]
                
                if _is_non_2_colorable(n, new_edges):
                    current_edges = new_edges
                    removed_any = True
        
        if len(current_edges) < best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
            # If we hit 51, we can stop early?
            # 51 is the known best. If we find 51, it's likely optimal or near-optimal.
            if best_count <= 51:
                break
                
    if best_edges is None:
        return {'n': 10, 'edges': all_edges_k10}
        
    return {'n': 10, 'edges': best_edges}
