<<<<<<< SEARCH
    # 3. If we haven't reached size 17, try backtracking for size 17
    # We know max size is likely 16 or 17.
    # If best_size < 17, we try to find a set of size 17.
    if best_size < 17 and time.time() < deadline - 5:
        # Give it 30 seconds to find a size 17 set
        time_limit = min(30, deadline - 5 - time.time())
        if time_limit > 1:
            result = _backtracking_search(universe, 17, time_limit)
            if result is not None:
                best_S = result
                best_size = 17

    return best_S
=======
    # 3. If we haven't reached size 17, try backtracking for size 17
    # We know max size is likely 16 or 17.
    # If best_size < 17, we try to find a set of size 17.
    if best_size < 17 and time.time() < deadline - 3:
        # Give it remaining time to find a size 17 set
        time_limit = min(35, deadline - 3 - time.time())
        if time_limit > 1:
            result = _backtracking_search(universe, 17, time_limit)
            if result is not None:
                best_S = result
                best_size = 17
    
    # 4. Final attempt: try to find size 18 if we have time
    if best_size < 18 and time.time() < deadline - 2:
        time_limit = min(15, deadline - 2 - time.time())
        if time_limit > 1:
            result = _backtracking_search(universe, 18, time_limit)
            if result is not None:
                best_S = result
                best_size = 18

    return best_S
>>>>>>> REPLACE

