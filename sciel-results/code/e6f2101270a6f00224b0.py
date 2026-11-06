    best_overall_points = set()
    best_overall_edges = -1
    
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 10: # Leave 10s margin
            break
        seed_pts = greedy_seed()
        pts, edges = local_search(seed_pts)
        if edges > best_overall_edges:
            best_overall_edges = edges
            best_overall_points = pts
            
    return [[a, b] for a, b in best_overall_points]
