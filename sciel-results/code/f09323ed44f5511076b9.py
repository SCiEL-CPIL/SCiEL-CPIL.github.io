def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    rank = 1
    
    U = [[1] * rank for _ in range(n * m)]   # 4 x 1
    V = [[1] * rank for _ in range(m * p)]   # 18 x 1
    W = [[1] * rank for _ in range(p * n)]   # 18 x 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify the rank-1 solution
    try:
        valid, diff = _verify_decomposition(U, V, W, n, m, p)
        if valid:
            return result
        else:
            # Fallback to rank-32
            return _construct_rank_32()
    except Exception:
        # Fallback to rank-32
        return _construct_rank_32()
