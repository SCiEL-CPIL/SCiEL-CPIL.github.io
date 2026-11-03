def construct():
    import time
    _t0 = time.time()
    _DEADLINE = 130.0

    n, m, p = 3, 4, 6
    terms = []  # Each term: (u_idx, v_idx, w_idx)

    # Strategy: Split the multiplication into two blocks based on the middle dimension (j)
    # A is 3x4, B is 4x6.
    # Split A into A1 (3x2) and A2 (3x2).
    # Split B into B1 (2x6) and B2 (2x6).
    # C = A1 * B1 + A2 * B2.
    # Each block is a 3x2x6 multiplication.
    # Schoolbook rank for 3x2x6 is 3*2*6 = 36.
    # Total rank = 36 + 36 = 72? No, wait.
    # A1 is 3x2, B1 is 2x6. Product is 3x6.
    # Rank of M_{3,2,6} is 3*2*6 = 36.
    # So total rank is 36 + 36 = 72.
    # This is the same as the full schoolbook.
    
    # Let's try splitting B instead?
    # B is 4x6. Split into B1 (4x3) and B2 (4x3).
    # C1 = A * B1 (3x3), C2 = A * B2 (3x3).
    # Rank of M_{3,4,3} is 3*4*3 = 36.
    # Total rank = 36 + 36 = 72.
    
    # It seems simple splitting doesn't reduce rank below schoolbook for these dimensions.
    # However, we can use a more efficient decomposition for M_{3,4,3} if available.
    # But for now, let's just implement the correct schoolbook to ensure it runs and scores > 0.
    # The previous failure was due to incorrect coefficients.
    
    # We will use the full schoolbook decomposition which is guaranteed to be correct.
    # Rank = 3 * 4 * 6 = 72.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                terms.append((u_idx, v_idx, w_idx))

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u_idx, v_idx, w_idx) in enumerate(terms):
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1

    if time.time() - _t0 > _DEADLINE:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
