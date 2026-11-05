<<<<<<< SEARCH
def _greedy_construct(order, time_limit):
    selected = []
    start = time.time()
    
    for x in order:
        if time.time() - start > time_limit:
            break
        if len(selected) >= 48:
            break
        
        if _fast_incremental_check(selected, x):
            C = selected + [x]
            # Full check
            if _is_valid(C):
                selected.append(x)
    
    return selected
=======
def _greedy_construct(order, time_limit):
    selected = []
    start = time.time()
    
    for x in order:
        if time.time() - start > time_limit:
            break
        if len(selected) >= 48:
            break
        
        if _fast_incremental_check(selected, x):
            C = selected + [x]
            # Full check
            if _is_valid(C):
                selected.append(x)
    
    return selected

def _greedy_construct_with_backtrack(order, time_limit):
    """
    Greedy with limited backtracking: if stuck, try removing last element
    and trying alternatives.
    """
    selected = []
    start = time.time()
    order_idx = 0
    
    while time.time() - start < time_limit and len(selected) < 48:
        # Try to add next elements until we fail
        progress = False
        while order_idx < len(order) and time.time() - start < time_limit and len(selected) < 48:
            x = order[order_idx]
            order_idx += 1
            if _fast_incremental_check(selected, x):
                C = selected + [x]
                if _is_valid(C):
                    selected.append(x)
                    progress = True
                    break
        
        if not progress:
            # Backtrack: remove last element and continue
            if len(selected) > 0:
                removed = selected.pop()
                # We'll re-try from the current order_idx
                # But we need to make sure we don't immediately re-add the same element
                # This is a simple heuristic
            else:
                break
    
    return selected
>>>>>>> REPLACE

