# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use 3 Strassen blocks for (0,1), (2,3), (4,5) and 1 trivial block for (6)
    # Rank = 7 * 3 + 4 = 25
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(k0):
        nonlocal col
        # Term 1: A00 * (B0k0 + B1k0+1) -> C0k0 + C1k0+1
        U[0][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k0 + 1][col] = 1
        W[k0 * n + 0][col] = 1
        W[(k0 + 1) * n + 1][col] = 1
        col += 1
        
        # Term 2: A10 * (B0k0 + B0k0+1) -> C1k0 - C1k0+1
        U[1 * m + 0][col] = 1
        V[0 * p + k0][col] = 1
        V[0 * p + k0 + 1][col] = 1
        W[k0 * n + 1][col] = 1
        W[(k0 + 1) * n + 1][col] = -1
        col += 1
        
        # Term 3: A01 * (B1k0 + B1k0+1) -> C0k0+1 + C1k0+1
        U[0 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[1 * p + k0 + 1][col] = 1
        W[(k0 + 1) * n + 0][col] = 1
        W[(k0 + 1) * n + 1][col] = 1
        col += 1
        
        # Term 4: A11 * (B0k0 - B1k0) -> C0k0 + C1k0
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # Term 5: A00 * (B0k0+1 - B1k0+1) -> C0k0+1
        U[0][col] = 1
        V[0 * p + k0 + 1][col] = 1
        V[1 * p + k0 + 1][col] = -1
        W[(k0 + 1) * n + 0][col] = 1
        col += 1
        
        # Term 6: A11 * (B1k0 + B1k0+1) -> -C0k0 + C1k0+1
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[1 * p + k0 + 1][col] = 1
        W[k0 * n + 0][col] = -1
        W[(k0 + 1) * n + 1][col] = 1
        col += 1
        
        # Term 7: A10 * (B0k0+1 - B0k0) -> C0k0
        U[1 * m + 0][col] = 1
        V[0 * p + k0 + 1][col] = 1
        V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        col += 1

    def add_trivial_block(k0):
        nonlocal col
        # A00 * B0k0 -> C0k0
        U[0][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # A01 * B1k0 -> C0k0
        U[0 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # A10 * B0k0 -> C1k0
        U[1 * m + 0][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # A11 * B1k0 -> C1k0
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

    # Apply blocks
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    add_trivial_block(6)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
