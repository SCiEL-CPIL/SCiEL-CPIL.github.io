import time

def construct():
    t0 = time.time()
    deadline = t0 + 135.0
    
    n, m, p = 2, 3, 8
    
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Strassen 2x2x2
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Rank 44 Construction
    # 4 blocks of 2x2x2 (rank 7 each) for a in {0,1}, k pairs (0,1),(2,3),(4,5),(6,7)
    # 1 block of 2x1x8 (rank 16) for a=2
    rank = 4 * 7 + 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4 blocks of 2x2x2 Strassen
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = k0 + 1
        for r in range(7):
            # U: i in {0,1}, a in {0,1}
            # U index: i*m + a
            for li in range(2):
                for la in range(2):
                    val = S_U[r][li * 2 + la]
                    if val != 0:
                        U[li * m + la][col] = val
            
            # V: a in {0,1}, k in {k0, k1}
            # V index: a*p + k
            for la in range(2):
                for lk in range(2):
                    val = S_V[r][la * 2 + lk]
                    if val != 0:
                        V[la * p + (k0 + lk)][col] = val
            
            # W: k in {k0, k1}, i in {0,1}
            # W index: k*n + i
            for lk in range(2):
                for li in range(2):
                    val = S_W[r][lk * 2 + li]
                    if val != 0:
                        W[(k0 + lk) * n + li][col] = val
            
            col += 1
        
        if (time.time() - t0) > deadline:
            return _schoolbook()
    
    # Part 2: Schoolbook for a=2
    # i in {0,1}, a=2, k in {0..7}
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
        
        if (time.time() - t0) > deadline:
            return _schoolbook()
    
    # Validate
    for i in range(n):
        for a in range(m):
            for k in range(p):
                acc = 0
                u_row = U[i * m + a]
                v_row = V[a * p + k]
                w_row = W[k * n + i]
                for c in range(rank):
                    if u_row[c] and v_row[c] and w_row[c]:
                        acc += u_row[c] * v_row[c] * w_row[c]
                if acc != 1:
                    return _schoolbook()
    
    if (time.time() - t0) > deadline:
        return _schoolbook()
    
    return {"U": U, "V": V, "W": W}
