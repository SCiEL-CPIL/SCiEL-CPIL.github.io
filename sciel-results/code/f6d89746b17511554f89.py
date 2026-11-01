"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen blocks for k in {0,1}, {2,3}, {4,5}, {6,7}
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1 = (A00 + A11) * (B00 + B11)
        # A00: i=0, j=0. A11: i=1, j=1
        U[0 * m + 0][col] += 1
        U[1 * m + 1][col] += 1
        # B00: j=0, k=k0. B11: j=1, k=k1
        V[0 * p + k0][col] += 1
        V[1 * p + k1][col] += 1
        # C00: k=k0, i=0. C11: k=k1, i=1
        W[k0 * n + 0][col] += 1
        W[k1 * n + 1][col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[1 * m + 0][col] += 1
        U[1 * m + 1][col] += 1
        V[0 * p + k0][col] += 1
        # C10: k=k0, i=1. C11: k=k1, i=1 (coeff -1)
        W[k0 * n + 1][col] += 1
        W[k1 * n + 1][col] += -1
        col += 1
        
        # M3 = A00 * (B10 - B11)
        U[0 * m + 0][col] += 1
        V[1 * p + k0][col] += 1
        V[1 * p + k1][col] += -1
        # C01: k=k1, i=0. C11: k=k1, i=1
        W[k1 * n + 0][col] += 1
        W[k1 * n + 1][col] += 1
        col += 1
        
        # M4 = A11 * (B01 - B00)
        U[1 * m + 1][col] += 1
        V[0 * p + k1][col] += 1
        V[0 * p + k0][col] += -1
        # C00: k=k0, i=0 (coeff -1). C10: k=k0, i=1
        W[k0 * n + 0][col] += -1
        W[k0 * n + 1][col] += 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[0 * m + 0][col] += 1
        U[0 * m + 1][col] += 1
        V[1 * p + k1][col] += 1
        # C00: k=k0, i=0
        W[k0 * n + 0][col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[1 * m + 0][col] += 1
        U[0 * m + 0][col] += -1
        V[0 * p + k0][col] += 1
        V[0 * p + k1][col] += 1
        # C11: k=k1, i=1
        W[k1 * n + 1][col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[0 * m + 1][col] += 1
        U[1 * m + 1][col] += -1
        V[1 * p + k0][col] += 1
        V[1 * p + k1][col] += 1
        # C00: k=k0, i=0 (coeff -1). C01: k=k1, i=0
        W[k0 * n + 0][col] += -1
        W[k1 * n + 0][col] += 1
        col += 1

    # Schoolbook block for k=8
    k = 8
    
    # A00 * B08 -> C08
    U[0 * m + 0][col] += 1
    V[0 * p + k][col] += 1
    W[k * n + 0][col] += 1
    col += 1
    
    # A01 * B18 -> C08
    U[0 * m + 1][col] += 1
    V[1 * p + k][col] += 1
    W[k * n + 0][col] += 1
    col += 1
    
    # A10 * B08 -> C18
    U[1 * m + 0][col] += 1
    V[0 * p + k][col] += 1
    W[k * n + 1][col] += 1
    col += 1
    
    # A11 * B18 -> C18
    U[1 * m + 1][col] += 1
    V[1 * p + k][col] += 1
    W[k * n + 1][col] += 1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
