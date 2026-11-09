<<<<<<< SEARCH
    # Final local search polish: try all single-element swaps and adds/removes
    A_mask = best_set_mask
    current_3ap_count = best_score
    improved = True
    polish_rounds = 0
    max_polish_rounds = 20
    
    while improved and time.time() < deadline and polish_rounds < max_polish_rounds:
        improved = False
        polish_rounds += 1
        
        # Try adding elements
        for x in range(1, N + 1):
            if not in_A(x):
                if is_valid_add(x):
                    delta = gain_add(x)
                    if delta > 0:
                        add_to_A(x)
                        current_3ap_count += delta
                        improved = True
        
        # Try removing elements
        for x in range(1, N + 1):
            if in_A(x):
                delta = -loss_remove(x)
                if delta > 0:
                    remove_from_A(x)
                    current_3ap_count += delta
                    improved = True
        
        # Try swaps
        active = [x for x in range(1, N + 1) if in_A(x)]
        inactive = [x for x in range(1, N + 1) if not in_A(x)]
        
        # Limit swap attempts for speed
        random.shuffle(active)
        random.shuffle(inactive)
        active_sample = active[:30]
        inactive_sample = inactive[:30]
        
        for x in active_sample:
            for y in inactive_sample:
                if is_valid_swap(x, y):
                    delta = delta_swap(x, y)
                    if delta > 0:
                        remove_from_A(x)
                        add_to_A(y)
                        current_3ap_count += delta
                        improved = True
=======
    # Final local search polish: try all single-element swaps and adds/removes
    A_mask = best_set_mask
    current_3ap_count = best_score
    improved = True
    polish_rounds = 0
    max_polish_rounds = 30
    
    while improved and time.time() < deadline - 2.0 and polish_rounds < max_polish_rounds:
        improved = False
        polish_rounds += 1
        
        # Try adding elements
        for x in range(1, N + 1):
            if not in_A(x):
                if is_valid_add(x):
                    delta = gain_add(x)
                    if delta > 0:
                        add_to_A(x)
                        current_3ap_count += delta
                        improved = True
        
        # Try removing elements
        for x in range(1, N + 1):
            if in_A(x):
                delta = -loss_remove(x)
                if delta > 0:
                    remove_from_A(x)
                    current_3ap_count += delta
                    improved = True
        
        # Try swaps - use larger samples
        active = [x for x in range(1, N + 1) if in_A(x)]
        inactive = [x for x in range(1, N + 1) if not in_A(x)]
        
        # Use larger swap samples for more thorough search
        random.shuffle(active)
        random.shuffle(inactive)
        active_sample = active[:50]
        inactive_sample = inactive[:50]
        
        for x in active_sample:
            for y in inactive_sample:
                if is_valid_swap(x, y):
                    delta = delta_swap(x, y)
                    if delta > 0:
                        remove_from_A(x)
                        add_to_A(y)
                        current_3ap_count += delta
                        improved = True
        
        # If time permits, do a second pass with even larger samples
        if time.time() < deadline - 3.0 and len(active) <= 100 and len(inactive) <= 100:
            for x in active:
                for y in inactive:
                    if is_valid_swap(x, y):
                        delta = delta_swap(x, y)
                        if delta > 0:
                            remove_from_A(x)
                            add_to_A(y)
                            current_3ap_count += delta
                            improved = True
>>>>>>> REPLACE
