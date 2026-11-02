"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank is 20 (part 1) + 5 * 7 (part 2) = 55
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: j=0 (index 0)
    # A_{i,0} * B_{0,k} * C_{i,k}
    c = 0
    for i in range(n):
        for k in range(p):
            # U index: i * m + 0
            U[i * m + 0][c] = 1
            # V index: 0 * p + k
            V[0 * p + k][c] = 1
            # W index: k * n + i
            W[k * n + i][c] = 1
            c += 1
            
    # Part 2: j=1,2 (indices 1, 2)
    # Split k into 5 blocks of size 2
    # For each block t, apply Strassen
    # Local A: rows i=0,1; cols j=1,2
    # Local B: rows j=1,2; cols k=2t, 2t+1
    # Local C: rows i=0,1; cols k=2t, 2t+1
    
    # Strassen coefficients
    # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    # M2: (A10+A11)B00    -> C01 += 1
    # M3: A00(B01-B11)    -> C10 += 1
    # M4: A11(B10-B00)    -> C00 += -1, C10 += 1
    # M5: (A00+A01)B11    -> C00 += 1
    # M6: (A10-A00)(B00+B01) -> C11 += 1
    # M7: (A01-A11)(B10+B11) -> C00 += -1, C01 += 1, C10 += 1 (Wait, check C10)
    # C10 = M3 + M4. Correct.
    # C11 = M1 - M2 + M3 + M6. Correct.
    # C00 = M1 - M4 + M5 - M7. Correct.
    # C01 = M2 + M7. Correct.
    
    # Define the linear forms for A and B for each M_l
    # A indices: (0,1), (0,2), (1,1), (1,2) -> global (i, j)
    # B indices: (1, 2t), (1, 2t+1), (2, 2t), (2, 2t+1) -> global (j, k)
    # C indices: (0, 2t), (0, 2t+1), (1, 2t), (1, 2t+1) -> global (i, k)
    
    # We will iterate t from 0 to 4
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1
        U[0 * m + 1][c] = 1
        U[1 * m + 2][c] = 1
        V[1 * p + k0][c] = 1
        V[2 * p + k1][c] = 1
        W[k0 * n + 0][c] = 1
        W[k1 * n + 1][c] = 1
        c += 1
        
        # M2
        U[1 * m + 1][c] = 1
        U[1 * m + 2][c] = 1
        V[1 * p + k0][c] = 1
        W[k1 * n + 0][c] = 1 # C01 is i=0, k=k1
        c += 1
        
        # M3
        U[0 * m + 1][c] = 1
        V[1 * p + k1][c] = 1
        V[2 * p + k1][c] = -1
        W[k0 * n + 1][c] = 1 # C10 is i=1, k=k0
        c += 1
        
        # M4
        U[1 * m + 2][c] = 1
        V[2 * p + k0][c] = 1
        V[1 * p + k0][c] = -1
        W[k0 * n + 0][c] = -1 # C00 coeff -1
        W[k0 * n + 1][c] = 1  # C10 coeff 1
        c += 1
        
        # M5
        U[0 * m + 1][c] = 1
        U[0 * m + 2][c] = 1
        V[2 * p + k1][c] = 1
        W[k0 * n + 0][c] = 1 # C00 coeff 1
        c += 1
        
        # M6
        U[1 * m + 1][c] = 1
        U[0 * m + 1][c] = -1
        V[1 * p + k0][c] = 1
        V[1 * p + k1][c] = 1
        W[k1 * n + 1][c] = 1 # C11 coeff 1
        c += 1
        
        # M7
        U[0 * m + 2][c] = 1
        U[1 * m + 2][c] = -1
        V[2 * p + k0][c] = 1
        V[2 * p + k1][c] = 1
        W[k0 * n + 0][c] = -1 # C00 coeff -1
        W[k1 * n + 0][c] = 1  # C01 coeff 1
        # Wait, C10 gets M7?
        # C10 = M3 + M4. No M7.
        # C01 = M2 + M7.
        # C00 = M1 - M4 + M5 - M7.
        # So M7 contributes to C00 and C01.
        # My W assignment:
        # W[k0 * n + 0] is C_{0, k0} -> C00. Correct.
        # W[k1 * n + 0] is C_{0, k1} -> C01. Correct.
        c += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
