"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(8):
        # Strassen terms for block b
        # Term 1: (a00+a11)(b00+b11)
        U[0][col] = 1
        U[3][col] = 1
        V[2*b][col] = 1
        V[16+2*b+1][col] = 1
        W[4*b][col] = 1
        W[4*b+3][col] = 1
        col += 1
        
        # Term 2: (a10+a11)b00
        U[2][col] = 1
        U[3][col] = 1
        V[2*b][col] = 1
        W[4*b+2][col] = 1
        W[4*b+3][col] = -1
        col += 1
        
        # Term 3: a00(b01-b11)
        U[0][col] = 1
        V[16+2*b][col] = 1
        V[16+2*b+1][col] = -1
        W[4*b+1][col] = 1
        W[4*b+3][col] = 1
        col += 1
        
        # Term 4: a11(b10-b00)
        U[3][col] = 1
        V[2*b+1][col] = 1
        V[2*b][col] = -1
        W[4*b][col] = 1
        W[4*b+2][col] = 1
        col += 1
        
        # Term 5: (a00+a01)b11
        U[0][col] = 1
        U[1][col] = 1
        V[16+2*b+1][col] = 1
        W[4*b][col] = -1
        W[4*b+1][col] = 1
        col += 1
        
        # Term 6: (a10-a00)(b00+b01)
        U[0][col] = -1
        U[2][col] = 1
        V[2*b][col] = 1
        V[16+2*b][col] = 1
        W[4*b+3][col] = 1
        col += 1
        
        # Term 7: (a01-a11)(b10+b11)
        U[1][col] = 1
        U[3][col] = -1
        V[2*b+1][col] = 1
        V[16+2*b+1][col] = 1
        W[4*b][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
