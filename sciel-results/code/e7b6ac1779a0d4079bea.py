def try_swap(best, deadline):
    """
    Try moving a comparator to a different position, then removing redundancies.
    """
    if time.time() > deadline - 5:
        return best
    
    best_len = len(best)
    attempts = 0
    max_attempts = 500
    
    while time.time() < deadline - 5 and attempts < max_attempts:
        attempts += 1
        
        # Pick a random comparator and a random new position
        pos = random.randint(0, len(best) - 1)
        new_pos = random.randint(0, len(best) - 1)
        
        if pos == new_pos:
            continue
            
        comp = best[pos]
        candidate = best[:pos] + best[pos+1:]
        # Adjust new_pos if it was after pos
        if new_pos > pos:
            new_pos -= 1
        candidate = candidate[:new_pos] + [comp] + candidate[new_pos:]
        
        # Try to remove comparators
        k = 0
        improved = True
        while k < len(candidate) and improved:
            improved = False
            if time.time() > deadline - 3:
                break
            test = candidate[:k] + candidate[k+1:]
            if is_sorted_fast(test):
                if is_sorted_network(test):
                    candidate = test
                    improved = True
                else:
                    k += 1
            else:
                k += 1
        
        if len(candidate) < best_len:
            best = candidate
            best_len = len(best)
            
    return best
