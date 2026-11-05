def construct():
    n, m, p = 2, 3, 10
    rank = 55
    
    # Initialize factor matrices
    # U: (n*m, rank) = (6, 55)
    # V: (m*p, rank) = (30, 55)
    # W: (p*n, rank) = (20, 55)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Contribution from A[:, 0] * B[0, :]
    # This is a 2x1 * 1x10 multiplication.
    # Rank is 2*10 = 20.
    # Indices:
    # U index: i * m + 0  (i in 0..1) -> 0, 3
    # V index: 0 * p + k  (k in 0..9) -> 0..9
    # W index: k * n + i  (k in 0..9, i in 0..1) -> 2k, 2k+1
    
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part 2: Contribution from A[:, 1:3] * B[1:3, :]
    # This is a 2x2 * 2x10 multiplication.
    # We can view this as 5 independent 2x2 * 2x2 multiplications (for column pairs of B).
    # Each 2x2 * 2x2 multiplication has rank 7 (Strassen).
    # Total rank for this part = 5 * 7 = 35.
    # Total rank = 20 + 35 = 55.
    
    # Indices for the 2x2 sub-matrix:
    # A sub-matrix rows i in {0, 1}, cols j in {1, 2}
    # U indices:
    # A[0, 1] -> i=0, j=1 -> 0*m + 1 = 1
    # A[0, 2] -> i=0, j=2 -> 0*m + 2 = 2
    # A[1, 1] -> i=1, j=1 -> 1*m + 1 = 4
    # A[1, 2] -> i=1, j=2 -> 1*m + 2 = 5
    
    # B sub-matrix rows j in {1, 2}, cols k in {0..9}
    # We process in pairs of columns: k0, k0+1 for k0 in {0, 2, 4, 6, 8}
    # V indices for a pair (k0, k0+1):
    # B[1, k0]   -> 1*p + k0
    # B[1, k0+1] -> 1*p + k0+1
    # B[2, k0]   -> 2*p + k0
    # B[2, k0+1] -> 2*p + k0+1
    
    # W indices for output C[0..1, k0..k0+1]:
    # C[0, k0]   -> k0*n + 0 = 2*k0
    # C[0, k0+1] -> (k0+1)*n + 0 = 2*(k0+1)
    # C[1, k0]   -> k0*n + 1 = 2*k0 + 1
    # C[1, k0+1] -> (k0+1)*n + 1 = 2*(k0+1) + 1
    
    # Strassen 7 multiplications for 2x2 * 2x2:
    # Let A' = [[A00, A01], [A10, A11]]
    # Let B' = [[B00, B01], [B10, B11]]
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Mapping to our indices:
    # A00 = U[1], A01 = U[2], A10 = U[4], A11 = U[5]
    
    for b in range(5):
        k0 = 2 * b
        
        # V indices
        v_b00 = 1 * p + k0
        v_b01 = 1 * p + k0 + 1
        v_b10 = 2 * p + k0
        v_b11 = 2 * p + k0 + 1
        
        # W indices
        w_c00 = k0 * n + 0
        w_c01 = (k0 + 1) * n + 0
        w_c10 = k0 * n + 1
        w_c11 = (k0 + 1) * n + 1
        
        # M1
        U[1][col] += 1
        U[5][col] += 1
        V[v_b00][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += 1
        W[w_c11][col] += 1
        col += 1
        
        # M2
        U[4][col] += 1
        U[5][col] += 1
        V[v_b00][col] += 1
        W[w_c10][col] += 1
        W[w_c11][col] -= 1
        col += 1
        
        # M3
        U[1][col] += 1
        V[v_b01][col] += 1
        V[v_b11][col] -= 1
        W[w_c01][col] += 1
        W[w_c11][col] += 1
        col += 1
        
        # M4
        U[5][col] += 1
        V[v_b10][col] += 1
        V[v_b00][col] -= 1
        W[w_c00][col] += 1
        W[w_c10][col] += 1
        col += 1
        
        # M5
        U[1][col] += 1
        U[2][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] -= 1
        W[w_c01][col] += 1
        col += 1
        
        # M6
        U[4][col] += 1
        U[1][col] -= 1
        V[v_b00][col] += 1
        V[v_b01][col] += 1
        W[w_c11][col] += 1
        col += 1
        
        # M7
        U[2][col] += 1
        U[5][col] -= 1
        V[v_b10][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
