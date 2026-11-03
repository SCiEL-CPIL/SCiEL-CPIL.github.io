DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Build one 2x3x2 block of rank 11, then tile over p/2 blocks.
    # Strassen terms for the 2x2x2 sub-problem (i in {0,1}, j in {0,1}, k in {0,1})
    # Each term: (U_entries, V_entries, W_entries)
    strassen = [
        # M1: (A00+A11)(B00+B11) -> C00, C11
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        # M2: (A10+A11)B00 -> C10
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1}),
        # M3: A00(B10-B11) -> C01
        ({(0,0):1}, {(1,0):1,(1,1):-1}, {(0,1):1}),
        # M4: A11(B01-B10) -> C00
        ({(1,1):1}, {(0,1):1,(1,0):-1}, {(0,0):1}),
        # M5: (A00+A01)B11 -> -C00, C01
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        # M6: (A10-A00)(B00+B10) -> C11
        ({(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}, {(1,1):1}),
        # M7: (A01-A11)(B01+B11) -> C00
        ({(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}, {(0,0):1}),
    ]
    # Trivial terms for j=2 (the third A column) for k in {0,1}
    trivial = []
    for i in range(2):
        for k in range(2):
            trivial.append(( {(i,2):1}, {(2,k):1}, {(i,k):1} ))
    
    block_terms = strassen + trivial  # 11 terms
    
    rank = (p // 2) * len(block_terms)  # 5 * 11 = 55
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for b in range(p // 2):
        for (Ue, Ve, We) in block_terms:
            for (i,j), val in Ue.items():
                U[i*m + j][col] = val
            for (j,k), val in Ve.items():
                k_global = 2*b + k
                V[j*p + k_global][col] = val
            for (i,k), val in We.items():
                k_global = 2*b + k
                W[k_global*n + i][col] = val
            col += 1
    
    return {"U": U, "V": V, "W": W}
