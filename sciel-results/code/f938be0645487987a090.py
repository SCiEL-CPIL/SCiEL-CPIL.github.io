"""Matrix multiplication 3,3,7 with rank-60 decomposition using Strassen for 2x2x2 blocks."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Strassen's rank-7 decomposition of M_{2,2,2}
    # Local indices: A is 2x2 (row*2+col), B is 2x2, C is 2x2
    strassen_terms = [
        # (u_local, v_local, w_local)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1=(a+d)(e+h)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # M2=(c+d)e
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),  # M3=a(f-h)
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),  # M4=d(g-e)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # M5=(a+b)h
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0]),  # M6=(c-a)(e+f)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),  # M7=(b-d)(g+h)
    ]
    
    # Schoolbook terms for M_{1,2,1} (rank 2)
    # A is 1x2 (row*2+col), B is 2x1, C is 1x1
    schoolbook_121 = [
        ([1, 0], [1, 0], [1]),  # a*e
        ([0, 1], [1, 0], [1]),  # b*f
    ]
    
    terms = []
    
    # Block 1: A rows 0-1, B rows 0-1, C rows 0-1 (2x2x2, Strassen)
    for u_loc, v_loc, w_loc in strassen_terms:
        u = [0] * (n * m)  # 9
        v = [0] * (m * p)  # 21
        w = [0] * (p * n)  # 21
        for idx, val in enumerate(u_loc):
            r, c = divmod(idx, 2)
            u[r * m + c] = val
        for idx, val in enumerate(v_loc):
            r, c = divmod(idx, 2)
            v[r * p + c] = val
        for idx, val in enumerate(w_loc):
            r, c = divmod(idx, 2)
            w[r * n + c] = val
        terms.append((u, v, w))
    
    # Block 2: A rows 0-1, B rows 0-1, C rows 2-6 (2x2x5)
    # Split into 2x2x2 (cols 2,3) + 2x2x1 (col 4) + 2x2x2 (cols 5,6)
    for u_loc, v_loc, w_loc in strassen_terms:
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for idx, val in enumerate(u_loc):
            r, c = divmod(idx, 2)
            u[r * m + c] = val
        for idx, val in enumerate(v_loc):
            r, c = divmod(idx, 2)
            v[r * p + (2 + c)] = val
        for idx, val in enumerate(w_loc):
            r, c = divmod(idx, 2)
            w[(2 + r) * n + c] = val
        terms.append((u, v, w))
    
    for u_loc, v_loc, w_loc in schoolbook_121:
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for idx, val in enumerate(u_loc):
            r, c = divmod(idx, 2)
            u[r * m + c] = val
        for idx, val in enumerate(v_loc):
            r, c = divmod(idx, 1)
            v[r * p + 4] = val
        w[4 * n + 0] = w_loc[0]
        w[4 * n + 1] = 0
        # Actually for M_{2,2,1}: C is 2x1, so w has 2 entries
        # w_loc has 2 entries for C rows 0,1
        w = [0] * (p * n)
        w[4 * n + 0] = w_loc[0]
        w[4 * n + 1] = w_loc[1]
        terms.append((u, v, w))
    
    for u_loc, v_loc, w_loc in strassen_terms:
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for idx, val in enumerate(u_loc):
            r, c = divmod(idx, 2)
            u[r * m + c] = val
        for idx, val in enumerate(v_loc):
            r, c = divmod(idx, 2)
            v[r * p + (5 + c)] = val
        for idx, val in enumerate(w_loc):
            r, c = divmod(idx, 2)
            w[(5 + r) * n + c] = val
        terms.append((u, v, w))
    
    # Block 3: A row 2, B rows 0-1, C rows 0-1 (1x2x2, schoolbook rank 4)
    for i in range(2):
        for k in range(2):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            u[2 * m + i] = 1
            v[i * p + k] = 1
            w[k * n + 2] = 1
            terms.append((u, v, w))
    
    # Block 4: A row 2, B rows 0-1, C rows 2-6 (1x2x5, schoolbook rank 10)
    for i in range(2):
        for k in range(2, 7):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            u[2 * m + i] = 1
            v[i * p + k] = 1
            w[k * n + 2] = 1
            terms.append((u, v, w))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u, v, w) in enumerate(terms):
        for i in range(n * m):
            U[i][col] = u[i]
        for i in range(m * p):
            V[i][col] = v[i]
        for i in range(p * n):
            W[i][col] = w[i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
