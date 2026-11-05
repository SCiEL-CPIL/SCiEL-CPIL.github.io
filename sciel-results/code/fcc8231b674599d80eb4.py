<<<<<<< SEARCH
    # Strategy: Start with {171,...,256}, try multiple random orders to add elements
    base_set = list(range(171, n + 1))
    best_S = list(base_set)
    best_size = len(best_S)
    
    random.seed(42)
    
    # Multiple random restarts with different addition orders
    for restart in range(60):
        if time.time() > deadline:
            break
        
        S = list(base_set)
        candidates = list(range(1, 171))
        
        if restart % 3 == 0:
            # Ascending order with some randomization
            random.shuffle(candidates)
            candidates.sort(key=lambda x: random.random())
        elif restart % 3 == 1:
            # Descending order with randomization
            random.shuffle(candidates)
            candidates.sort(key=lambda x: -random.random())
        else:
            # Pure random order
            random.shuffle(candidates)
        
        S = greedy_add(S, candidates, candidates)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
    
    # Local search: try to add more elements
    best_S = local_search(best_S)
    best_size = len(best_S)
    
    # Swap search: try to improve via swaps
    best_S = swap_search(best_S)
    best_size = len(best_S)
    
    # Final verification
    S_final = sorted(best_S)
    if not check_valid(S_final):
        # Fallback to known valid set
        return list(range(171, 257))
    
    return S_final
=======
    # Strategy: Try multiple starting sets with different thresholds
    random.seed(42)
    best_S = []
    best_size = 0
    
    # Different starting thresholds
    thresholds = [171, 165, 160, 155, 150, 145, 140, 135, 130, 125, 120]
    
    for threshold in thresholds:
        if time.time() > deadline:
            break
        
        base_set = list(range(threshold, n + 1))
        S = list(base_set)
        
        # Local search on base
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
    
    # Multiple random restarts from best base
    base_threshold = 171
    base_set = list(range(base_threshold, n + 1))
    
    for restart in range(80):
        if time.time() > deadline:
            break
        
        S = list(base_set)
        candidates = list(range(1, base_threshold))
        
        if restart % 4 == 0:
            # Random order
            random.shuffle(candidates)
        elif restart % 4 == 1:
            # Descending
            candidates.sort(reverse=True)
        elif restart % 4 == 2:
            # Ascending
            candidates.sort()
        else:
            # Weighted random (prefer larger numbers first)
            random.shuffle(candidates)
            candidates.sort(key=lambda x: random.random() ** 0.5)
        
        S = greedy_add(S, candidates, candidates)
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
    
    # Swap search: try to improve via swaps
    best_S = swap_search(best_S)
    best_size = len(best_S)
    
    # Multi-swap search for further improvement
    best_S = multi_swap_search(best_S, max_swaps=2)
    best_size = len(best_S)
    
    # Final local search polish
    best_S = local_search(best_S)
    best_size = len(best_S)
    
    # Final verification
    S_final = sorted(best_S)
    if not check_valid(S_final):
        # Fallback to known valid set
        return list(range(171, 257))
    
    return S_final
>>>>>>> REPLACE

