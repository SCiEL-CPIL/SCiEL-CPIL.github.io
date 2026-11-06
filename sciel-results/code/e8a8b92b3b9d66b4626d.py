def _local_search(s_set, time_limit=10.0):
    start_time = time.time()
    best_set = set(s_set)
    current_set = set(s_set)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try to add a new element
        for candidate in range(1, N + 1):
            if candidate in current_set:
                continue
            new_set = current_set | {candidate}
            if _is_valid_fast(new_set):
                current_set = new_set
                if len(current_set) > len(best_set):
                    best_set = set(current_set)
                improved = True
                break # Restart scan
        
        if improved:
            continue

        # 2. Try multiple random swaps
        swaps_tried = 0
        while swaps_tried < 10 and time.time() - start_time < time_limit:
            swaps_tried += 1
            if not current_set:
                break
            x = random.choice(list(current_set))
            temp_set = current_set - {x}
            
            added = False
            # Try to add a random y
            candidates = [y for y in range(1, N+1) if y not in temp_set and y != x]
            if not candidates:
                break
            y = random.choice(candidates)
            new_set = temp_set | {y}
            if _is_valid_fast(new_set):
                current_set = new_set
                if len(current_set) > len(best_set):
                    best_set = set(current_set)
                added = True
                break # Restart outer loop if improved
            
            # If not added, we can either revert or keep the removal?
            # Keeping removal reduces size, which is bad. So revert.
            current_set.add(x)
            
    return best_set
