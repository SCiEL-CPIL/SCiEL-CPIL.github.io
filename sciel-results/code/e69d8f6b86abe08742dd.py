"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Strassen part for j=0,1
    for p_idx in range(4):
        k0 = 2 * p_idx
        k1 = 2 * p_idx + 1
        
        # P1: (A00 + A11)(B0k0 + B1k1)
        U[0][r] = 1; U[4][r] = 1
        V[k0][r] = 1; V[8 + k1][r] = 1
        W[2 * k0][r] = 1; W[2 * k1 + 1][r] = 1
        r += 1
        
        # P2: (A10 + A11)B0k0
        U[3][r] = 1; U[4][r] = 1
        V[k0][r] = 1
        W[2 * k0 + 1][r] = 1
        r += 1
        
        # P3: A00(B0k1 - B1k1)
        U[0][r] = 1
        V[k1][r] = 1; V[8 + k1][r] = -1
        W[2 * k1][r] = 1; W[2 * k1 + 1][r] = 1
        r += 1
        
        # P4: A11(B1k0 - B0k0)
        U[4][r] = 1
        V[8 + k0][r] = 1; V[k0][r] = -1
        W[2 * k0][r] = 1; W[2 * k0 + 1][r] = 1
        r += 1
        
        # P5: (A00 + A01)B1k1
        U[0][r] = 1; U[1][r] = 1
        V[8 + k1][r] = 1
        W[2 * k1][r] = 1; W[2 * k0][r] = -1
        r += 1
        
        # P6: (A10 - A00)(B0k0 + B0k1)
        U[3][r] = 1; U[0][r] = -1
        V[k0][r] = 1; V[k1][r] = 1
        W[2 * k0][r] = 1
        r += 1
        
        # P7: (A01 - A11)(B1k0 + B1k1)
        U[1][r] = 1; U[4][r] = -1
        V[8 + k0][r] = 1; V[8 + k1][r] = 1
        W[2 * k1 + 1][r] = 1
        r += 1

    # Schoolbook part for j=2
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
