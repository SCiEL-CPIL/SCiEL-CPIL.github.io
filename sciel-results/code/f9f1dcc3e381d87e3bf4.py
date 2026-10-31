<<<<<<< SEARCH
    # Phase 2: Local search — remove redundant blocks
    blocks = _local_search(blocks, v, k, t, deadline, rng)
    
    return blocks
=======
    # Phase 2: Fast local search — remove redundant blocks using coverage counts
    blocks = _local_search_fast(blocks, v, k, t, deadline, rng)
    
    # Phase 3: Swap-based local search to escape local minima
    if time.time() < deadline - 5:
        blocks = _swap_local_search(blocks, v, k, t, deadline - 5, rng)
    
    # Final pass: try to remove more redundant blocks after swaps
    if time.time() < deadline - 2:
        blocks = _local_search_fast(blocks, v, k, t, deadline - 2, rng)
    
    return blocks
>>>>>>> REPLACE
