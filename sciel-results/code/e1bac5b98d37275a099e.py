def _construct_rank_18(n, m, p):
    if n != 3 or m != 3 or p != 4:
        return None
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j][r] = 1
            for k in range(3):
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    for i in range(n):
        for j in range(m):
            r = 9 + i * m + j
            U[i * m + j][r] = 1
            V[j * p + 3][r] = 1
            W[3 * n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    best_result = _construct_rank_18(n, m, p)
    
    if best_result is not None:
        if _validate(best_result['U'], best_result['V'], best_result['W'], n, m, p, 18):
            # Try to find better (rank < 18)
            for rank in range(12, 18):
                if time.time() > deadline:
                    break
                rng = random.Random(rank * 1000 + 42)
                res = _try_fixed_U_V_search(rank, n, m, p, deadline, rng, max_iter=2000)
                if res is not None:
                    return res
            return best_result
        else:
            # Construction failed validation, fall back
            best_result = _schoolbook(n, m, p)
            
    # If construction failed or n,m,p different, use old search
    try:
        result = _try_systematic_ranks(n, m, p, deadline)
        if result is not None:
            return result
    except Exception:
        pass
        
    if best_result is None:
        best_result = _schoolbook(n, m, p)
        
    return best_result
