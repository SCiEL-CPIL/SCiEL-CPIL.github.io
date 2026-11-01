"""Matrix multiplication 2x3 by 3x8 using block-wise Strassen decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen for A_top (2x2) * B_top (2x2)
        U[0][col] = 1; U[4][col] = 1
        V[k0][col] = 1; V[p+k1][col] = 1
        W[k0][col] = 1; W[n+k1][col] = 1
        col += 1
        
        U[3][col] = 1; U[4][col] = 1
        V[k0][col] = 1
        W[n+k0][col] = 1; W[n+k1][col] = -1
        col += 1
        
        U[0][col] = 1
        V[k1][col] = 1; V[p+k1][col] = -1
        W[k1][col] = 1; W[n+k1][col] = 1
        col += 1
        
        U[4][col] = 1
        V[p+k0][col] = 1; V[k0][col] = -1
        W[k0][col] = 1; W[n+k0][col] = 1
        col += 1
        
        U[0][col] = 1; U[1][col] = 1
        V[p+k1][col] = 1
        W[k0][col] = -1; W[k1][col] = 1
        col += 1
        
        U[3][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[k1][col] = 1
        W[n+k1][col] = 1
        col += 1
        
        U[1][col] = 1; U[4][col] = -1
        V[p+k0][col] = 1; V[p+k1][col] = 1
        W[k0][col] = 1
        col += 1
        
        # Outer product for A_bottom (1x2) * B_top (2x2)
        U[2][col] = 1
        V[k0][col] = 1
        W[k0][col] = 1
        col += 1
        
        U[2][col] = 1
        V[k1][col] = 1
        W[k1][col] = 1
        col += 1
        
        U[5][col] = 1
        V[k0][col] = 1
        W[n+k0][col] = 1
        col += 1
        
        U[5][col] = 1
        V[k1][col] = 1
        W[n+k1][col] = 1
        col += 1
        
        # Outer product for A_bottom * B1
        U[2][col] = 1
        V[p+k0][col] = 1
        W[k0][col] = 1
        col += 1
        
        U[2][col] = 1
        V[p+k1][col] = 1
        W[k1][col] = 1
        col += 1
        
        U[5][col] = 1
        V[p+k0][col] = 1
        W[n+k0][col] = 1
        col += 1
        
        U[5][col] = 1
        V[p+k1][col] = 1
        W[n+k1][col] = 1
        col += 1
        
        # Outer product for A_bottom * B2
        U[2][col] = 1
        V[2*p+k0][col] = 1
        W[k0][col] = 1
        col += 1
        
        U[2][col] = 1
        V[2*p+k1][col] = 1
        W[k1][col] = 1
        col += 1
        
        U[5][col] = 1
        V[2*p+k0][col] = 1
        W[n+k0][col] = 1
        col += 1
        
        U[5][col] = 1
        V[2*p+k1][col] = 1
        W[n+k1][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
