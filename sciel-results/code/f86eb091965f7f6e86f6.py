DIMENSIONS = [2, 3, 5]

def construct():
    import time
    start_time = time.time()
    DEADLINE = 145.0  # 5 seconds safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 5
    rank = 29
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 29
    V = [[0.0] * rank for _ in range(m * p)]  # 15 x 29
    W = [[0.0] * rank for _ in range(p * n)]  # 10 x 29
    
    col = 0
    
    # Strassen block: A1 (2x2) * B1a (2x2)
    strassen_terms = [
        # (A_coeffs: [(i,j,coeff)], B_coeffs: [(j,k,coeff)], C_coeffs: [(i,k,coeff)])
        # M1 = (A00+A11)*(B00+B11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A10+A11)*B00
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3 = A00*(B01-B11)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1)]),
        # M4 = A11*(B10-B00)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (A00+A01)*B11
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (A10-A00)*(B00+B01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (A01-A11)*(B10+B11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,1,-1)]),
    ]
    
    for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
        for (i, j, coeff) in a_coeffs:
            U[i*m+j][col] += coeff
        for (j, k, coeff) in b_coeffs:
            V[j*p+k][col] += coeff
        for (i, k, coeff) in c_coeffs:
            W[k*n+i][col] += coeff
        col += 1
    
    # A1*B1b: A1 (2x2) * B1b (2x3), k in {2,3,4}
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    
    # A2*B2: A2 (2x1) * B2 (1x5)
    for i in range(2):
        for k in range(5):
            U[i*m+2][col] = 1.0
            V[2*p+k][col] = 1.0
            W[k*n+i][col] = 1.0
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Should not happen for this fixed construction
        pass
    
    return {"U": U, "V": V, "W": W}
