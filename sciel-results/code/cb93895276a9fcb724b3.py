<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Generate Schoolbook Decomposition
    U, V, W = _build_schoolbook_decomposition(n, m, p)
    
    # 2. Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: should not happen
        pass
        
    # 3. Attempt Rank Reduction
    # Try to find a lower-rank decomposition
    result = _search_lower_rank(n, m, p, deadline)
    if result is not None:
        U, V, W = result
    
    return {"U": U, "V": V, "W": W}
=======
def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # 1. Generate Schoolbook Decomposition
    U, V, W = _build_schoolbook_decomposition(n, m, p)
    
    # 2. Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: should not happen
        pass
        
    # 3. Attempt Rank Reduction
    # Try to find a lower-rank decomposition
    result = _search_lower_rank(n, m, p, deadline)
    if result is not None:
        U, V, W = result
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
