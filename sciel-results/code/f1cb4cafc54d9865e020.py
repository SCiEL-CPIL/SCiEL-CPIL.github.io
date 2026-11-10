def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_edges = None
    best_n = 0
    best_len = float('inf')
    
    # Prioritize smaller n first as they have fewer edges to start with, 
    # but might require more edges to be non-2-colorable? 
    # Actually, larger n allows sparser graphs.
    # But search space is larger.
    # Let's try n=9, 10, 11.
    
    for n in [9, 10, 11]:
        if time.time() > deadline - 15:
            break
            
        all_edges = list(combinations(range(n), 5))
        
        # Quick check: is K_n^5 non-2-colorable?
        # If yes, we can proceed.
        if is_2_colorable(n, all_edges):
            continue
            
        # Greedy removal with limited trials
        for trial in range(3):
            if time.time() > deadline - 5:
                break
                
            random.seed(42 + n * 100 + trial)
            trial_edges = all_edges[:]
            random.shuffle(trial_edges)
            
            i = 0
            while i < len(trial_edges) and time.time() < deadline - 5:
                test_edges = trial_edges[:i] + trial_edges[i+1:]
                if not is_2_colorable(n, test_edges):
                    trial_edges = test_edges
                else:
                    i += 1
            
            if len(trial_edges) < best_len:
                best_edges = trial_edges[:]
                best_n = n
                best_len = len(trial_edges)
                
    if best_edges is None:
        n = 9
        all_edges = list(combinations(range(n), 5))
        return {'n': n, 'edges': [list(e) for e in all_edges]}
        
    return {'n': best_n, 'edges': [list(e) for e in best_edges]}
