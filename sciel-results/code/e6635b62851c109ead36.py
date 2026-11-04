<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # Try to find a rank-24 decomposition for 3x3x3, then extend to rank 27
    rng = random.Random(42)
    base_result = _search_rank_base(3, 3, 3, 24, deadline, rng, max_iter=5000)
    if base_result is not None:
        U_base, V_base, W_base = base_result
        U, V, W = _extend_333_to_334(U_base, V_base, W_base, 24)
        return {"U": U, "V": V, "W": W}
        
    # Fallback: Search for rank 27 directly on 3x3x4
    rng2 = random.Random(123)
    result = _search_rank_base(n, m, p, 27, deadline, rng2, max_iter=5000)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
        
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
=======
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # Try to find a rank-24 decomposition for 3x3x3, then extend to rank 27
    rng = random.Random(42)
    base_result = _search_rank_base(3, 3, 3, 24, deadline, rng, max_iter=5000)
    if base_result is not None:
        U_base, V_base, W_base = base_result
        U, V, W = _extend_333_to_334(U_base, V_base, W_base, 24)
        # Sanitize to ensure JSON serializability
        U = [[int(x) if isinstance(x, Fraction) and x.denominator == 1 else float(x) for x in row] for row in U]
        V = [[int(x) if isinstance(x, Fraction) and x.denominator == 1 else float(x) for x in row] for row in V]
        W = [[int(x) if isinstance(x, Fraction) and x.denominator == 1 else float(x) for x in row] for row in W]
        return {"U": U, "V": V, "W": W}
        
    # Fallback: Search for rank 27 directly on 3x3x4
    rng2 = random.Random(123)
    result = _search_rank_base(n, m, p, 27, deadline, rng2, max_iter=5000)
    if result is not None:
        U, V, W = result
        # Sanitize to ensure JSON serializability
        U = [[int(x) if isinstance(x, Fraction) and x.denominator == 1 else float(x) for x in row] for row in U]
        V = [[int(x) if isinstance(x, Fraction) and x.denominator == 1 else float(x) for x in row] for row in V]
        W = [[int(x) if isinstance(x, Fraction) and x.denominator == 1 else float(x) for x in row] for row in W]
        return {"U": U, "V": V, "W": W}
        
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
