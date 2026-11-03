<<<<<<< SEARCH
    best_blocks = None
    best_len = float('inf')

    # Run multiple greedy passes with different strategies
    attempt = 0
    while time.time() < deadline - 5.0:
        if attempt % 2 == 0:
            selected = _greedy_pass_fast(attempt)
        else:
            selected = _greedy_pass(attempt)
        
        if selected is None:
            break

        # Remove redundant blocks
        selected = _remove_redundant(selected)

        if len(selected) < best_len:
            best_len = len(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]

        # Early stop if we've found a very good solution
        if best_len <= 340:
            break

        attempt += 1

    # Fallback if no solution found
    if best_blocks is None:
        selected = _greedy_pass_fast(0)
        if selected is not None:
            selected = _remove_redundant(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]
        else:
            best_blocks = []

    return best_blocks
=======
    best_blocks = None
    best_len = float('inf')

    # Run multiple greedy passes with different strategies
    attempt = 0
    while time.time() < deadline - 8.0:
        if attempt % 3 == 0:
            selected = _greedy_pass_fast(attempt)
        elif attempt % 3 == 1:
            selected = _greedy_pass(attempt)
        else:
            # Mixed: urgency greedy with different seed
            selected = _greedy_pass_fast(attempt * 7 + 13)

        if selected is None:
            break

        # Remove redundant blocks
        selected = _remove_redundant(selected)

        # Local search polish
        selected = _local_search_polish(selected)

        if len(selected) < best_len:
            best_len = len(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]

        # Early stop if we've found a very good solution
        if best_len <= 335:
            break

        attempt += 1

    # Fallback if no solution found
    if best_blocks is None:
        selected = _greedy_pass_fast(0)
        if selected is not None:
            selected = _remove_redundant(selected)
            selected = _local_search_polish(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]
        else:
            best_blocks = []

    return best_blocks
>>>>>>> REPLACE
