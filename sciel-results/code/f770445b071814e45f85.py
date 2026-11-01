"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 5 groups * 7 Strassen multiplications = 35
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Mapping of A indices (i*m+j)
    # A00: 0, A01: 1, A10: 2, A11: 3
    
    # For each group of 2 columns in B (and C)
    for g in range(p // 2):
        k1 = 2 * g
        k2 = 2 * g + 1
        
        # V indices for B
        # B0,k1: 0*10 + k1 = k1
        # B1,k1: 1*10 + k1 = 10 + k1
        # B0,k2: 0*10 + k2 = k2
        # B1,k2: 1*10 + k2 = 10 + k2
        
        # W indices for C
        # C0,k1: k1*2 + 0 = 2*k1
        # C1,k1: k1*2 + 1 = 2*k1 + 1
        # C0,k2: k2*2 + 0 = 2*k2
        # C1,k2: k2*2 + 1 = 2*k2 + 1
        
        # Step 0: M1
        col = g * 7 + 0
        U[0][col] += 1  # A00
        U[3][col] += 1  # A11
        V[k1][col] += 1  # B0,k1
        V[10 + k2][col] += 1  # B1,k2
        W[2 * k1][col] += 1  # C0,k1
        W[2 * k2 + 1][col] += 1  # C1,k2
        
        # Step 1: M2
        col = g * 7 + 1
        U[2][col] += 1  # A10
        U[3][col] += 1  # A11
        V[k1][col] += 1  # B0,k1
        W[2 * k1 + 1][col] += 1  # C1,k1
        W[2 * k2 + 1][col] -= 1  # C1,k2
        
        # Step 2: M3
        col = g * 7 + 2
        U[0][col] += 1  # A00
        V[10 + k1][col] += 1  # B1,k1
        V[10 + k2][col] -= 1  # B1,k2
        W[2 * k2][col] += 1  # C0,k2
        W[2 * k2 + 1][col] += 1  # C1,k2
        
        # Step 3: M4
        col = g * 7 + 3
        U[3][col] += 1  # A11
        V[k2][col] += 1  # B0,k2
        V[k1][col] -= 1  # B0,k1
        W[2 * k1][col] += 1  # C0,k1
        W[2 * k1 + 1][col] += 1  # C1,k1
        
        # Step 4: M5
        col = g * 7 + 4
        U[0][col] += 1  # A00
        U[1][col] += 1  # A01
        V[10 + k2][col] += 1  # B1,k2
        W[2 * k1][col] -= 1  # C0,k1
        W[2 * k2][col] += 1  # C0,k2
        
        # Step 5: M6
        col = g * 7 + 5
        U[2][col] += 1  # A10
        U[0][col] -= 1  # A00
        V[k1][col] += 1  # B0,k1
        V[10 + k1][col] += 1  # B1,k1
        W[2 * k2 + 1][col] += 1  # C1,k2
        
        # Step 6: M7
        col = g * 7 + 6
        U[1][col] += 1  # A01
        U[3][col] -= 1  # A11
        V[10 + k2][col] += 1  # B1,k2
        V[k2][col] += 1  # B0,k2
        W[2 * k1][col] += 1  # C0,k1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
