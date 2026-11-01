def construct():
    import time
    start = time.time()
    DEADLINE = 110  # safety margin before 120s (or 150s) kill

    n, m, p = 2, 4, 5
    # U: (n*m, rank) = (8, rank), row = i*m + j = i*4 + j
    # V: (m*p, rank) = (20, rank), row = j*p + k = j*5 + k
    # W: (p*n, rank) = (10, rank), row = k*n + i = k*2 + i

    U = {}  # row -> {col: val}
    V = {}
    W = {}
    next_col = 0

    # Strassen steps for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    STRASSE = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    def add_24x42(a_cols, b_cols):
        nonlocal next_col
        for part in range(2):
            for (X, Y, P) in STRASSE:
                col = next_col
                next_col += 1
                for (i, j), c in X.items():
                    r = i*4 + a_cols[part*2 + j]
                    U.setdefault(r, {})[col] = U.get(r, {}).get(col, 0) + c
                for (a, b), c in Y.items():
                    r = a_cols[part*2 + a]*5 + b_cols[b]
                    V.setdefault(r, {})[col] = V.get(r, {}).get(col, 0) + c
                for (i, b), c in P.items():
                    r = b_cols[b]*2 + i
                    W.setdefault(r, {})[col] = W.get(r, {}).get(col, 0) + c

    def add_24x41(a_cols, b_col):
        nonlocal next_col
        # Schoolbook for 2x4x1
        # C_i = sum_{l=0}^3 A_{i,l} B_{l,2}
        # Split into two 2x2x1 parts
        for part in range(2):
            for l_local in range(2):
                for i in range(2):
                    col = next_col
                    next_col += 1
                    # U: A_{i, l}
                    r_u = i*4 + a_cols[part*2 + l_local]
                    U.setdefault(r_u, {})[col] = U.get(r_u, {}).get(col, 0) + 1
                    # V: B_{l, 2}
                    r_v = a_cols[part*2 + l_local]*5 + b_col
                    V.setdefault(r_v, {})[col] = V.get(r_v, {}).get(col, 0) + 1
                    # W: C_{i, 2}
                    r_w = b_col*2 + i
                    W.setdefault(r_w, {})[col] = W.get(r_w, {}).get(col, 0) + 1

    # Sub-problems
    add_24x42([0,1,2,3], [3,4])  # C2 (cols 3,4)
    add_24x42([0,1,2,3], [0,1])  # C3a (cols 0,1)
    add_24x41([0,1,2,3], 2)      # C3b (col 2)

    rank = next_col
    # Convert to lists
    U_list = [[0]*rank for _ in range(8)]
    V_list = [[0]*rank for _ in range(20)]
    W_list = [[0]*rank for _ in range(10)]
    for r, cols in U.items():
        for c, val in cols.items():
            U_list[r][c] = val
    for r, cols in V.items():
        for c, val in cols.items():
            V_list[r][c] = val
    for r, cols in W.items():
        for c, val in cols.items():
            W_list[r][c] = val

    return {"U": U_list, "V": V_list, "W": W_list}
