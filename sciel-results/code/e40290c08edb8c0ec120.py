"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition: 5 pairs of columns (Strassen 2x2x2) + 1 single column (Schoolbook 2x2x1)
    # Strassen rank 7 per pair, Schoolbook rank 4 for single.
    # Total rank = 5 * 7 + 4 = 39.
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    r = 0
    
    # Process 5 pairs of columns
    for _ in range(5):
        k0 = col_idx
        k1 = col_idx + 1
        col_idx += 2
        
        # Strassen 2x2x2 decomposition
        # M1 = (a11+a22)(b11+b22)
        U[0][r] = 1; U[3][r] = 1
        V[k0][r] = 1; V[2+k1][r] = 1
        W[k0*n + 0][r] = 1; W[k1*n + 1][r] = 1
        W[k0*n + 1][r] = 1; W[k1*n + 0][r] = -1
        r += 1
        
        # M2 = (a21+a22)b11
        U[2][r] = 1; U[3][r] = 1
        V[k0][r] = 1
        W[k0*n + 1][r] = 1; W[k1*n + 1][r] = -1
        r += 1
        
        # M3 = a11(b21-b22)
        U[0][r] = 1
        V[2+k1][r] = -1; V[2+k0][r] = 1
        W[k0*n + 0][r] = 1; W[k1*n + 0][r] = 1
        r += 1
        
        # M4 = a22(b12-b11)
        U[3][r] = 1
        V[k1][r] = 1; V[k0][r] = -1
        W[k0*n + 0][r] = 1; W[k1*n + 1][r] = 1
        r += 1
        
        # M5 = (a11+a12)b22
        U[0][r] = 1; U[1][r] = 1
        V[2+k1][r] = 1
        W[k0*n + 0][r] = -1; W[k1*n + 0][r] = 1
        r += 1
        
        # M6 = (a21-a11)(b11+b12)
        U[2][r] = 1; U[0][r] = -1
        V[k0][r] = 1; V[k1][r] = 1
        W[k1*n + 1][r] = 1
        r += 1
        
        # M7 = (a12-a22)(b21+b22)
        U[1][r] = 1; U[3][r] = -1
        V[2+k0][r] = 1; V[2+k1][r] = 1
        W[k1*n + 0][r] = 1
        r += 1
        
    # Process the last single column (index 10)
    k0 = col_idx
    # Schoolbook 2x2x1
    # C11 = a11*b1
    U[0][r] = 1; V[k0][r] = 1; W[k0*n + 0][r] = 1
    r += 1
    # C12 = a12*b2 (Note: B is 2x1, so b2 is row 1 of B)
    # Wait, B is 2x11. Row 0 is b1, Row 1 is b2.
    # V index is j*p + k.
    # For single column k0, V[0*p+k0] is b1, V[1*p+k0] is b2.
    U[1][r] = 1; V[p + k0][r] = 1; W[k0*n + 0][r] = 1
    r += 1
    # C21 = a21*b1
    U[2][r] = 1; V[k0][r] = 1; W[k0*n + 1][r] = 1
    r += 1
    # C22 = a22*b2
    U[3][r] = 1; V[p + k0][r] = 1; W[k0*n + 1][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
