def _local_search(S, counts, time_limit=100.0):
    """
    Local search to improve the set size.
    S is a list, counts is the sum-count dict (defaultdict).
    """
    S_set = set(S)
    S_sorted = sorted(S)
    start_time = time.time()
    
    # Ensure counts is a defaultdict
    if not isinstance(counts, defaultdict):
        counts = defaultdict(int, counts)
        
    candidates = [x for x in range(1, 1001) if x not in S_set]
    
    best_size = len(S)
    best_S = list(S)
    best_counts = defaultdict(int, counts)
    
    def update_best():
        nonlocal best_size, best_S, best_counts
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
            best_counts = defaultdict(int, counts)

    # Phase 1: Greedy Add
    improved = True
    while improved:
        improved = False
        for x in candidates[:]:  # iterate over a copy
            if time.time() - start_time > time_limit:
                break
            if _can_add(S_sorted, counts, x):
                S.append(x)
                S_set.add(x)
                S_sorted.append(x)
                S_sorted.sort()
                for s in S_sorted:
                    counts[x + s] += 1
                counts[2 * x] += 1
                candidates.remove(x)
                improved = True
                update_best()
        if time.time() - start_time > time_limit:
            break
    
    # Phase 2: Randomized Swaps with Heuristics
    random.seed(42)
    attempts = 0
    max_attempts = 200000 # Increased attempts
    while time.time() - start_time < time_limit and attempts < max_attempts:
        attempts += 1
        if not S or not candidates:
            break
            
        # Heuristic: Pick a random element to remove, but bias towards those 
        # that are involved in "full" sums (count == 2) to free up space?
        # Or just random for now, but fix the state bug.
        
        idx = random.randint(0, len(S) - 1)
        remove_x = S[idx]
        add_x = random.choice(candidates)
        
        # Save state correctly
        old_counts = defaultdict(int, counts)
        old_S = list(S)
        old_S_set = set(S_set)
        old_S_sorted = list(S_sorted)
        
        # Remove remove_x
        S.remove(remove_x)
        S_set.remove(remove_x)
        S_sorted.remove(remove_x)
        for s in S_sorted:
            counts[remove_x + s] -= 1
        counts[2 * remove_x] -= 1
        # Clean up zeros to keep dict small? Optional, but good for memory if it grows
        # Not strictly necessary for correctness with defaultdict
        
        # Try to add add_x
        if _can_add(S_sorted, counts, add_x):
            S.append(add_x)
            S_set.add(add_x)
            S_sorted.append(add_x)
            S_sorted.sort()
            for s in S_sorted:
                counts[add_x + s] += 1
            counts[2 * add_x] += 1
            candidates.remove(add_x)
            # Size is same, but structure changed. 
            # Check if this allows adding more immediately?
            # We'll rely on the outer loop or a quick check.
            # Let's try a quick greedy add here
            temp_improved = True
            while temp_improved and time.time() - start_time < time_limit:
                temp_improved = False
                for y in candidates[:]:
                    if _can_add(S_sorted, counts, y):
                        S.append(y)
                        S_set.add(y)
                        S_sorted.append(y)
                        S_sorted.sort()
                        for s in S_sorted:
                            counts[y + s] += 1
                        counts[2 * y] += 1
                        candidates.remove(y)
                        temp_improved = True
                        update_best()
                    if time.time() - start_time > time_limit:
                        break
        else:
            # Restore state
            S = old_S
            S_set = old_S_set
            S_sorted = old_S_sorted
            counts = old_counts
            
    # Phase 3: Final Greedy Pass
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for x in candidates[:]:
            if time.time() - start_time > time_limit:
                break
            if _can_add(S_sorted, counts, x):
                S.append(x)
                S_set.add(x)
                S_sorted.append(x)
                S_sorted.sort()
                for s in S_sorted:
                    counts[x + s] += 1
                counts[2 * x] += 1
                candidates.remove(x)
                improved = True
                update_best()
    
    return best_S
