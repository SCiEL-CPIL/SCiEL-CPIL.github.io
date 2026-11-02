"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 5 blocks of 2x2
    # Block l uses columns 2*l and 2*l+1 of B
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    for l in range(5):
        k1 = 2 * l
        k2 = 2 * l + 1
        base_col = 7 * l
        
        # M1: (a+d)(x+w) -> C11(+1), C22(+1)
        c = base_col + 0
        U[0][c] = 1; U[3][c] = 1
        V[0*p + k1][c] = 1; V[1*p + k2][c] = 1
        W[k1*n + 0][c] = 1; W[k2*n + 1][c] = 1
        
        # M2: (c+d)x -> C21(+1), C22(-1)
        c = base_col + 1
        U[2][c] = 1; U[3][c] = 1
        V[0*p + k1][c] = 1
        W[k1*n + 1][c] = 1; W[k2*n + 1][c] = -1
        
        # M3: a(y-w) -> C12(+1), C22(+1)
        c = base_col + 2
        U[0][c] = 1
        V[0*p + k2][c] = 1; V[1*p + k2][c] = -1
        W[k2*n + 0][c] = 1; W[k2*n + 1][c] = 1
        
        # M4: d(z-x) -> C11(+1), C21(+1)
        c = base_col + 3
        U[3][c] = 1
        V[1*p + k1][c] = 1; V[0*p + k1][c] = -1
        W[k1*n + 0][c] = 1; W[k1*n + 1][c] = 1
        
        # M5: (a+b)w -> C11(-1), C12(+1)
        c = base_col + 4
        U[0][c] = 1; U[1][c] = 1
        V[1*p + k2][c] = 1
        W[k1*n + 0][c] = -1; W[k2*n + 0][c] = 1
        
        # M6: (c-a)(x+y) -> C22(+1)
        c = base_col + 5
        U[2][c] = 1; U[0][c] = -1
        V[0*p + k1][c] = 1; V[0*p + k2][c] = 1
        W[k2*n + 1][c] = 1
        
        # M7: (b-d)(z+w) -> C11(+1)
        c = base_col + 6
        U[1][c] = 1; U[3][c] = -1
        V[1*p + k1][c] = 1; V[1*p + k2][c] = 1
        W[k1*n + 0][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
