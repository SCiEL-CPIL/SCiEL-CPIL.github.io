"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 25 decomposition using Strassen for 2x2 blocks and schoolbook for 2x1 block
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for columns k0, k1
    def add_strassen_block(k0, k1):
        nonlocal col
        # M1: (a00+a11)(b00+b11)
        U[0][col] += 1; U[3][col] += 1
        V[0*p + k0][col] += 1; V[1*p + k1][col] += 1
        W[k0*n + 0][col] += 1; W[k1*n + 1][col] += 1
        col += 1
        
        # M2: (a10+a11)b00
        U[2][col] += 1; U[3][col] += 1
        V[0*p + k0][col] += 1
        W[k0*n + 1][col] += 1; W[k1*n + 1][col] -= 1
        col += 1
        
        # M3: a00(b10-b11)
        U[0][col] += 1
        V[1*p + k0][col] += 1; V[1*p + k1][col] -= 1
        W[k1*n + 0][col] += 1; W[k1*n + 1][col] += 1
        col += 1
        
        # M4: a11(b01-b00)
        U[3][col] += 1
        V[0*p + k1][col] += 1; V[0*p + k0][col] -= 1
        W[k0*n + 0][col] += 1; W[k0*n + 1][col] += 1
        col += 1
        
        # M5: (a00+a01)b11
        U[0][col] += 1; U[1][col] += 1
        V[1*p + k1][col] += 1
        W[k0*n + 0][col] -= 1; W[k1*n + 0][col] += 1
        col += 1
        
        # M6: (a10-a00)(b00+b10)
        U[2][col] += 1; U[0][col] -= 1
        V[0*p + k0][col] += 1; V[1*p + k0][col] += 1
        W[k1*n + 1][col] += 1
        col += 1
        
        # M7: (a01-a11)(b01+b11)
        U[1][col] += 1; U[3][col] -= 1
        V[0*p + k1][col] += 1; V[1*p + k1][col] += 1
        W[k0*n + 0][col] += 1
        col += 1

    # Block 1: cols 0, 1
    add_strassen_block(0, 1)
    # Block 2: cols 2, 3
    add_strassen_block(2, 3)
    # Block 3: cols 4, 5
    add_strassen_block(4, 5)
    
    # Block 4: col 6 (2x1 schoolbook)
    # N1: a00 b06
    U[0][col] += 1
    V[0*p + 6][col] += 1
    W[6*n + 0][col] += 1
    col += 1
    
    # N2: a01 b16
    U[1][col] += 1
    V[1*p + 6][col] += 1
    W[6*n + 0][col] += 1
    col += 1
    
    # N3: a10 b06
    U[2][col] += 1
    V[0*p + 6][col] += 1
    W[6*n + 1][col] += 1
    col += 1
    
    # N4: a11 b16
    U[3][col] += 1
    V[1*p + 6][col] += 1
    W[6*n + 1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
