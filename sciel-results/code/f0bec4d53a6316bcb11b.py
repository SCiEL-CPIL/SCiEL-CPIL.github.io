def construct():
    import time
    _t0 = time.time()
    _DEADLINE = 140.0

    n, m, p = 3, 4, 6
    terms = []  # Each term: (u_dict, v_dict, w_dict)
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        """Add Strassen decomposition for A[i1:i2+1, j1:j2+1] * B[j1:j2+1, k1:k2+1] -> C[i1:i2+1, k1:k2+1]"""
        a = (i1, j1)
        b = (i1, j2)
        c = (i2, j1)
        d = (i2, j2)
        e = (j1, k1)
        f = (j1, k2)
        g = (j2, k1)
        h = (j2, k2)
        
        C11 = (i1, k1)
        C12 = (i1, k2)
        C21 = (i2, k1)
        C22 = (i2, k2)
        
        terms.append((
            {a: 1, d: 1},
            {e: 1, h: 1},
            {C11: 1, C22: 1}
        ))
        terms.append((
            {c: 1, d: 1},
            {e: 1},
            {C21: 1, C22: -1}
        ))
        terms.append((
            {a: 1},
            {f: 1, h: -1},
            {C12: 1, C22: 1}
        ))
        terms.append((
            {d: 1},
            {g: 1, e: -1},
            {C11: 1, C21: 1}
        ))
        terms.append((
            {a: 1, b: 1},
            {h: 1},
            {C11: -1, C12: 1}
        ))
        terms.append((
            {c: 1, a: -1},
            {e: 1, f: 1},
            {C22: 1}
        ))
        terms.append((
            {b: 1, d: -1},
            {g: 1, h: 1},
            {C11: 1}
        ))
    
    def add_schoolbook_term(i, j, k):
        terms.append((
            {(i, j): 1},
            {(j, k): 1},
            {(i, k): 1}
        ))
    
    # Strategy: Split rows into (0,1) using Strassen and (2) using schoolbook
    # Split columns into pairs: (0,1), (2,3), (4,5)
    # For each column pair, use Strassen for rows 0-1 (7 terms) and schoolbook for row 2 (4 terms)
    for (k1, k2) in [(0, 1), (2, 3), (4, 5)]:
        # Strassen for rows 0-1, cols j1=0,j2=1 -> k1,k2
        add_strassen_block(0, 1, 0, 1, k1, k2)
        # Schoolbook for row 2, cols 0,1 -> k1,k2
        for j in [0, 1]:
            for k in [k1, k2]:
                add_schoolbook_term(2, j, k)
        
        # Strassen for rows 0-1, cols j1=2,j2=3 -> k1,k2
        add_strassen_block(0, 1, 2, 3, k1, k2)
        # Schoolbook for row 2, cols 2,3 -> k1,k2
        for j in [2, 3]:
            for k in [k1, k2]:
                add_schoolbook_term(2, j, k)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = coeff
    
    if time.time() - _t0 > _DEADLINE:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
