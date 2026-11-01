"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using Strassen's algorithm for each 2x2 block of the 12 columns
    # There are 6 blocks of size 2x2
    # Each 2x2x2 multiplication has rank 7
    # Total rank = 6 * 7 = 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(6):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen terms for block b
        # M1 = (A00 + A11)(B0k0 + B1k1)
        U[0][col] = 1 # A00
        U[3][col] = 1 # A11
        V[k0][col] = 1 # B0k0
        V[12 + k1][col] = 1 # B1k1
        W[k0 * 2 + 0][col] = 1 # C0k0
        W[k1 * 2 + 1][col] = 1 # C1k1
        col += 1
        
        # M2 = (A10 + A11)B0k0
        U[2][col] = 1 # A10
        U[3][col] = 1 # A11
        V[k0][col] = 1 # B0k0
        W[k0 * 2 + 1][col] = 1 # C1k0
        col += 1
        
        # M3 = A00(B0k1 - B1k1)
        U[0][col] = 1 # A00
        V[12 + k0][col] = 0 # B0k0 (not used in V for this term? Wait)
        # V index for B is j*p + k.
        # B0k1 is j=0, k=k1 -> index 0*12 + k1 = k1
        # B1k1 is j=1, k=k1 -> index 1*12 + k1 = 12 + k1
        V[k1][col] = 1 # B0k1
        V[12 + k1][col] = -1 # B1k1
        W[k1 * 2 + 0][col] = 1 # C0k1
        W[k1 * 2 + 1][col] = 1 # C1k1
        col += 1
        
        # M4 = A11(B1k0 - B0k0)
        U[3][col] = 1 # A11
        V[12 + k0][col] = 1 # B1k0
        V[k0][col] = -1 # B0k0
        W[k0 * 2 + 0][col] = 1 # C0k0
        W[k0 * 2 + 1][col] = 1 # C1k0
        col += 1
        
        # M5 = (A00 + A01)B1k1
        U[0][col] = 1 # A00
        U[1][col] = 1 # A01
        V[12 + k1][col] = 1 # B1k1
        W[k0 * 2 + 0][col] = -1 # C0k0
        W[k1 * 2 + 0][col] = 1 # C0k1
        col += 1
        
        # M6 = (A10 - A00)(B0k0 + B0k1)
        U[2][col] = 1 # A10
        U[0][col] = -1 # A00
        V[k0][col] = 1 # B0k0
        V[k1][col] = 1 # B0k1
        W[k1 * 2 + 1][col] = 1 # C1k1
        col += 1
        
        # M7 = (A01 - A11)(B1k0 + B1k1)
        U[1][col] = 1 # A01
        U[3][col] = -1 # A11
        V[12 + k0][col] = 1 # B1k0
        V[12 + k1][col] = 1 # B1k1
        W[k0 * 2 + 0][col] = 1 # C0k0
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
