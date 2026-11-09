<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Try approach 1: 4 blocks of 3x3 with rank 23 each -> total rank 92
    try:
        U1, V1, W1 = _build_6x6_from_3x3_blocks()
        rank1 = len(U1[0]) if U1 else 0
        valid1, err1 = _verify_decomposition(U1, V1, W1, n, m, p)
        if valid1 and rank1 < best_rank:
            best_rank = rank1
            best_result = {"U": U1, "V": V1, "W": W1}
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        if best_result:
            return best_result
        # Fallback to known valid solution
        U2, V2, W2 = _build_6x6_from_2x2_blocks_strassen()
        return {"U": U2, "V": V2, "W": W2}
    
    # Try approach 2: 9 blocks of 2x2 with Strassen -> total rank 189 (baseline)
    try:
        U2, V2, W2 = _build_6x6_from_2x2_blocks_strassen()
        rank2 = len(U2[0]) if U2 else 0
        valid2, err2 = _verify_decomposition(U2, V2, W2, n, m, p)
        if valid2 and rank2 < best_rank:
            best_rank = rank2
            best_result = {"U": U2, "V": V2, "W": W2}
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        if best_result:
            return best_result
        # Absolute fallback
        U2, V2, W2 = _build_6x6_from_2x2_blocks_strassen()
        return {"U": U2, "V": V2, "W": W2}
    
    if best_result:
        return best_result
    
    # Final fallback
    U2, V2, W2 = _build_6x6_from_2x2_blocks_strassen()
    return {"U": U2, "V": V2, "W": W2}
=======
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: 3x3 grid of 2x2 Strassen blocks -> rank 189
    try:
        U1, V1, W1 = _build_6x6_strassen_tiling()
        rank1 = len(U1[0]) if U1 else 0
        valid1, err1 = _verify_decomposition(U1, V1, W1, n, m, p)
        if valid1 and rank1 < best_rank:
            best_rank = rank1
            best_result = {"U": U1, "V": V1, "W": W1}
    except Exception:
        pass
    
    if time.time() > deadline - 20:
        if best_result:
            return best_result
        # Fallback
        U2, V2, W2 = _build_6x6_from_2x2_blocks_strassen()
        return {"U": U2, "V": V2, "W": W2}
    
    # If we have a valid result, return it
    if best_result:
        return best_result
    
    # Final fallback
    U2, V2, W2 = _build_6x6_from_2x2_blocks_strassen()
    return {"U": U2, "V": V2, "W": W2}
>>>>>>> REPLACE
