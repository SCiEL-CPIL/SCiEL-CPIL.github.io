"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 * 7 = 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # A is 2x2, B is 2x2, C is 2x2
    # Indices: A[i,j], B[j,k], C[i,k]
    # Strassen products M1..M7
    
    # We process 4 blocks of 2x2 matrices
    for q in range(4):
        c0 = 2 * q
        c1 = 2 * q + 1
        
        # Term 0: M1 = (A00+A11)(B00+B11)
        r = 7 * q + 0
        U[0][r] = 1  # A00
        U[3][r] = 1  # A11
        V[c0][r] = 1    # B00 (row 0, col c0)
        V[8 + c1][r] = 1 # B11 (row 1, col c1)
        W[2 * c0][r] = 1     # C00 (k=c0, i=0)
        W[2 * c1 + 1][r] = 1 # C11 (k=c1, i=1)
        
        # Term 1: M2 = (A10+A11)B00
        r = 7 * q + 1
        U[2][r] = 1  # A10
        U[3][r] = 1  # A11
        V[c0][r] = 1    # B00
        W[2 * c0 + 1][r] = 1   # C10 (k=c0, i=1)
        W[2 * c1 + 1][r] = -1  # C11 (k=c1, i=1)
        
        # Term 2: M3 = A00(B01-B11)
        r = 7 * q + 2
        U[0][r] = 1  # A00
        V[c1][r] = 1    # B01
        V[8 + c1][r] = -1 # B11
        W[2 * c1][r] = 1     # C01 (k=c1, i=0)
        W[2 * c1 + 1][r] = 1 # C11 (k=c1, i=1)
        
        # Term 3: M4 = A11(B10-B00)
        r = 7 * q + 3
        U[3][r] = 1  # A11
        V[8 + c0][r] = 1  # B10
        V[c0][r] = -1     # B00
        W[2 * c0][r] = 1     # C00 (k=c0, i=0)
        W[2 * c0 + 1][r] = 1 # C10 (k=c0, i=1)
        
        # Term 4: M5 = (A00+A01)B11
        r = 7 * q + 4
        U[0][r] = 1  # A00
        U[1][r] = 1  # A01
        V[8 + c1][r] = 1 # B11
        W[2 * c0][r] = -1   # C00 (k=c0, i=0)
        W[2 * c1][r] = 1    # C01 (k=c1, i=0)
        
        # Term 5: M6 = (A10-A00)(B00+B01)
        r = 7 * q + 5
        U[2][r] = 1   # A10
        U[0][r] = -1  # A00
        V[c0][r] = 1  # B00
        V[c1][r] = 1  # B01
        W[2 * c1 + 1][r] = 1 # C11 (k=c1, i=1)
        
        # Term 6: M7 = (A01-A11)(B10+B11)
        r = 7 * q + 6
        U[1][r] = 1   # A01
        U[3][r] = -1  # A11
        V[8 + c0][r] = 1 # B10
        V[8 + c1][r] = 1 # B11
        W[2 * c0][r] = 1   # C00 (k=c0, i=0)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
