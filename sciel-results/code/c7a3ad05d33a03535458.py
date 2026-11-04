<<<<<<< SEARCH
    # Phase 2: Backtracking construction (try to do better)
    if time.time() < deadline - 30.0:
        bt_rows = _construct_with_backtracking(n_rows, k, deadline - 30.0)
        bt_scope = _scope(bt_rows)
        if bt_scope < best_scope:
            best_rows = bt_rows
            best_scope = bt_scope
=======
    # Phase 2: Backtracking construction with constraint-degree heuristic
    if time.time() < deadline - 30.0:
        bt_rows = _construct_with_backtracking(n_rows, k, deadline - 30.0)
        bt_scope = _scope(bt_rows)
        if bt_scope < best_scope:
            best_rows = bt_rows
            best_scope = bt_scope
    
    # Phase 2b: Second backtracking attempt with different random seed
    if time.time() < deadline - 25.0:
        rng2 = random.Random(42)
        bt_rows2 = _construct_with_backtracking(n_rows, k, deadline - 25.0, rng=rng2)
        bt_scope2 = _scope(bt_rows2)
        if bt_scope2 < best_scope:
            best_rows = bt_rows2
            best_scope = bt_scope2
>>>>>>> REPLACE
