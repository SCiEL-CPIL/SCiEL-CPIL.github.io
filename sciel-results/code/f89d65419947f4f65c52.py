def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    n = P['n']
    base_edges = P['edges']
    
    # 1. Construct a valid initial solution: Star from 12
    # Connect 12 to all even nodes except 12, and all odd nodes except those adjacent to 12?
    # Actually, the previous star was:
    # Even: 0, 2, 4, 6, 8
    # Odd: 15, 17, 19, 21, 23
    # This was valid.
    
    initial_edges = [
        [0, 12], [2, 12], [4, 12], [6, 12], [8, 12],
        [15, 12], [17, 12], [19, 12], [21, 12], [23, 12]
    ]
    
    if not _valid_solution(initial_edges, base_edges, n):
        # Fallback: Full star from 12 to all nodes at distance > 2
        # This might create triangles? 
        # Star from 12 to all v where |v-12|>2.
        # If we connect 12 to 10 and 12 to 14.
        # 10-11-12-13-14.
        # Triangle? 10-12 (added), 12-14 (added), 10-14? No.
        # 10-11 (path), 11-12 (path), 12-10 (added). Triangle 10-11-12?
        # Edge (10,12) added. Edge (10,11) path. Edge (11,12) path.
        # Yes, triangle 10-11-12.
        # So we CANNOT connect 12 to 10 or 14.
        # We must skip neighbors of 12 (11, 13) and their neighbors (10, 14) if we want to be safe?
        # No, just skip distance 2.
        # So valid star from 12: connect to all v such that |v-12| > 2.
        # But we must also ensure no other triangles.
        # Since all edges are incident to 12, the only triangles would be (12, v, w) where v-w is path edge.
        # So we cannot have both (12, v) and (12, v+1).
        # So we must pick an independent set of neighbors of 12?
        # Neighbors of 12 in path are 11, 13.
        # If we add (12, 11), we cannot add (12, 10) or (12, 12) (no) or (12, 14)?
        # (12, 11) and (12, 10): Triangle 10-11-12?
        # 10-11 (path), 11-12 (path), 12-10 (added). Yes.
        # So we cannot add (12, 10) if we have (12, 11).
        # Since 11 is a path neighbor, we usually don't add (12, 11) because it's distance 1 (duplicate).
        # So we skip 11, 13.
        # We can add (12, 10)?
        # If we add (12, 10), we cannot add (12, 9) or (12, 11) or (12, 12) or (12, 14)?
        # (12, 10) and (12, 14)?
        # 10-11-12-13-14.
        # Triangle? 10-12 (added), 12-14 (added), 10-14? No.
        # 10-11 (path), 11-12 (path), 12-10 (added). No, 11-12 is path, 12-10 is added.
        # Is there a triangle 10-11-12?
        # Edges: (10,11) path, (11,12) path, (12,10) added.
        # Yes, triangle 10-11-12.
        # So we CANNOT add (12, 10).
        # Similarly, we CANNOT add (12, 14).
        # So for a star from 12, we can only connect to vertices at distance >= 3?
        # If dist 3: (12, 9).
        # Triangle? 9-10-11-12.
        # Edges: (9,10) path, (10,11) path, (11,12) path, (12,9) added.
        # Triangle 9-10-12? No, 10-12 not edge.
        # Triangle 9-12-11? No, 9-11 not edge.
        # So (12, 9) is safe.
        
        # So a valid star from 12 connects to all v with |v-12| >= 3.
        fallback = []
        for v in range(n):
            if v != 12 and abs(v - 12) >= 3:
                fallback.append([min(12, v), max(12, v)])
        
        if _valid_solution(fallback, base_edges, n):
            initial_edges = fallback
        else:
            # If even that fails, return empty (invalid) or a known valid small set?
            # The previous 10-edge star was valid. Use it.
            initial_edges = [
                [0, 12], [2, 12], [4, 12], [6, 12], [8, 12],
                [15, 12], [17, 12], [19, 12], [21, 12], [23, 12]
            ]

    best_edges = [e[:] for e in initial_edges]
    best_count = len(best_edges)
    
    # 2. Local Search: Try to remove edges
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_edges)):
            trial = best_edges[:i] + best_edges[i+1:]
            if _valid_solution(trial, base_edges, n):
                best_edges = trial
                best_count = len(best_edges)
                improved = True
                break  # Restart loop as list changed
    
    # 3. Local Search: Try to swap edges
    # Generate candidates
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if abs(u - v) <= 2:
                continue
            if (u, v) in [(e[0], e[1]) for e in best_edges] or (v, u) in [(e[0], e[1]) for e in best_edges]:
                continue
            candidates.append([u, v])
            
    # Try swapping one edge in best with one candidate
    # This is expensive. Limit candidates.
    # Only try if we have time.
    if time.time() < deadline - 5:
        # Sort candidates by potential utility (distance)
        candidates.sort(key=lambda e: abs(e[0] - e[1]), reverse=True)
        
        for i in range(len(best_edges)):
            if time.time() > deadline:
                break
            for cand in candidates[:20]:
                if time.time() > deadline:
                    break
                # Check if cand is already in best
                if [min(cand[0], cand[1]), max(cand[0], cand[1])] in [[min(e[0], e[1]), max(e[0], e[1])] for e in best_edges]:
                    continue
                trial = best_edges[:i] + best_edges[i+1:] + [cand]
                if _valid_solution(trial, base_edges, n):
                    # If valid, we have same count. 
                    # Does this help? Maybe it allows further removals.
                    # Let's accept it and try to remove again.
                    best_edges = trial
                    improved = True
                    break
            if improved:
                break
        
        # Run removal again after swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_edges)):
                trial = best_edges[:i] + best_edges[i+1:]
                if _valid_solution(trial, base_edges, n):
                    best_edges = trial
                    best_count = len(best_edges)
                    improved = True
                    break

    # Convert to proper types
    result = []
    for e in best_edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
