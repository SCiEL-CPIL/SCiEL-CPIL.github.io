"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 4x5 multiplication into two 2x5 multiplications.
    # Part 1: j=0,1. Part 2: j=2,3.
    # Each 2x5 part has rank 3*5 = 15.
    # Total rank = 30.
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Winograd 2x2x5 decomposition
    # For j=0,1:
    # u0 = A_i0, u1 = A_i1
    # v0 = B_0k, v1 = B_1k
    # Terms: (u0+u1)(v0+v1), u0*v0, u1*v1
    # Coeffs in W: 1, -1, -1
    
    # Block 1: j=0,1
    for k in range(p):
        # Term 1: (A_i0 + A_i1) * (B_0k + B_1k)
        r = k
        for i in range(n):
            U[i * m + 0][r] = 1
            U[i * m + 1][r] = 1
        V[0 * p + k][r] = 1
        V[1 * p + k][r] = 1
        for i in range(n):
            W[k * n + i][r] = 1
            
        # Term 2: A_i0 * B_0k
        r = p + k
        for i in range(n):
            U[i * m + 0][r] = 1
        V[0 * p + k][r] = 1
        for i in range(n):
            W[k * n + i][r] = -1
            
        # Term 3: A_i1 * B_1k
        r = 2 * p + k
        for i in range(n):
            U[i * m + 1][r] = 1
        V[1 * p + k][r] = 1
        for i in range(n):
            W[k * n + i][r] = -1

    # Block 2: j=2,3
    for k in range(p):
        # Term 4: (A_i2 + A_i3) * (B_2k + B_3k)
        r = 3 * p + k
        for i in range(n):
            U[i * m + 2][r] = 1
            U[i * m + 3][r] = 1
        V[2 * p + k][r] = 1
        V[3 * p + k][r] = 1
        for i in range(n):
            W[k * n + i][r] = 1
            
        # Term 5: A_i2 * B_2k
        r = 4 * p + k
        for i in range(n):
            U[i * m + 2][r] = 1
        V[2 * p + k][r] = 1
        for i in range(n):
            W[k * n + i][r] = -1
            
        # Term 6: A_i3 * B_3k
        r = 5 * p + k
        for i in range(n):
            U[i * m + 3][r] = 1
        V[3 * p + k][r] = 1
        for i in range(n):
            W[k * n + i][r] = -1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
