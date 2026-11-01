import time

DIMENSIONS = [3, 3, 4]

def _target_tensor(n, m, p):
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                key = (i*m+j, j*p+k, k*n+i)
                T[key] = T.get(key, 0) + 1
    return T

def _from_columns(cols, n, m, p, rank):
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for r, (Ac, Bc, Cc) in enumerate(cols):
        for (i, j), v in Ac.items():
            U[i*m+j][r] += v
        for (j, k), v in Bc.items():
            V[j*p+k][r] += v
        for (i, k), v in Cc.items():
            W[k*n+i][r] += v
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    target = _target_tensor(n, m, p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j; b = j*p+k; c = k*n+i
                s = 0
                for r in range(rank):
                    s += U[a][r]*V[b][r]*W[c][r]
                if abs(s - target.get((a, b, c), 0)) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def _build_rank24(n, m, p):
    """
    Try to construct a rank-24 decomposition for 3x3x4 matrix multiplication.
    
    Strategy: Decompose into four 3x3x2 blocks (A * B[:,0:2] and A * B[:,2:4]),
    each with a rank-12 schoolbook, but share 12 intermediate products.
    
    Actually, let me try a different approach: use a block decomposition where
    we treat the 3x3 matrix as having structure that allows sharing.
    
    Let me try: split A into A1 (3x1) and A2 (3x2), B into B1 (1x4) and B2 (2x4).
    C = A1*B1 + A2*B2
    A1*B1: 3x1 * 1x4 = 3x4, rank 12 (schoolbook)
    A2*B2: 3x2 * 2x4 = 3x4, rank 24 (schoolbook)
    Total: 36, no improvement.
    
    Let me try yet another approach. For 3x3x4, I'll use the following:
    - Use Strassen on a 2x2x4 sub-block (rows 0-1 of A, all of B)
    - Handle row 2 of A separately
    """
    cols = []
    
    def add_col(Ac, Bc, Cc):
        cols.append((dict(Ac), dict(Bc), dict(Cc)))
    
    # Part 1: Rows 0-1 of A (2x3) times B (3x4) -> C[0:2, 0:4] (2x4)
    # Split B into B_left (3x2) and B_right (3x2)
    # Use Strassen-like for 2x3 x 3x2 -> 2x2
    
    # For 2x3 x 3x2: A is 2x3, B is 3x2, C is 2x2
    # Schoolbook rank: 2*3*2 = 12
    # Can we do better? Let's use a direct construction.
    
    # A = [[a00, a01, a02],
    #      [a10, a11, a12]]
    # B = [[b00, b01],
    #      [b10, b11],
    #      [b20, b21]]
    # C = [[c00, c01],
    #      [c10, c11]]
    
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    
    # Let's use 7 products via Strassen-like for this case:
    # m1 = (a00+a10)*(b00+b10)  -> contributes to c00, c01, c10, c11
    # m2 = a02*(b20+b21)        -> contributes to c00, c01
    # m3 = (a11-a01)*(b10+b11)  -> contributes to c01, c10, c11
    # m4 = (a00+a01)*(b00+b01)  -> contributes to c00, c01
    # m5 = (a10+a11)*(b10+b11)  -> contributes to c10, c11
    # m6 = a01*(b10-b00)        -> contributes to c00, c01
    # m7 = a11*(b11-b01)        -> contributes to c10, c11
    # 
    # This is getting complicated. Let me just use schoolbook for the 2x3x3x2 part.
    
    # 2x3 x 3x2 schoolbook (12 terms):
    for i in range(2):
        for j in range(3):
            for k in range(2):
                add_col({(i,j):1}, {(j,k):1}, {(i,k):1})
    
    # Part 2: Row 2 of A (1x3) times B (3x4) -> C[2, 0:4] (1x4)
    # Schoolbook: 1*3*4 = 12 terms
    for j in range(3):
        for k in range(4):
            add_col({(2,j):1}, {(j,k):1}, {(2,k):1})
    
    rank = len(cols)
    # This gives 12 + 12 = 24
    return _from_columns(cols, n, m, p, rank)

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    DEADLINE = 115.0
    
    # Try rank-24 construction
    U, V, W = _build_rank24(n, m, p)
    rank24 = len(U[0])
    if _verify(U, V, W, n, m, p, rank24) and (time.time() - t0) < DEADLINE:
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    if time.time() - t0 < DEADLINE:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    return {"U": [[0]], "V": [[0]], "W": [[0]]}
