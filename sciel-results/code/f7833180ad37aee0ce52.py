"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    products = [
        (0, 0, 0, 0, 0, 0),   # X0 * B00 -> C[:,0:2]
        (0, 2, 0, 0, 0, 0),   # X1 * B10 -> C[:,0:2]
        (0, 0, 0, 2, 0, 2),   # X0 * B01 -> C[:,2:4]
        (0, 2, 0, 2, 0, 2),   # X1 * B11 -> C[:,2:4]
    ]
    
    for (arow0, acol0, brow0, bcol0, crown0, ccol0) in products:
        A00 = arow0 * m + acol0
        A01 = arow0 * m + acol0 + 1
        A10 = (arow0+1) * m + acol0
        A11 = (arow0+1) * m + acol0 + 1
        B00 = brow0 * p + bcol0
        B01 = brow0 * p + bcol0 + 1
        B10 = (brow0+1) * p + bcol0
        B11 = (brow0+1) * p + bcol0 + 1
        C00 = ccol0 * n + crown0
        C01 = (ccol0+1) * n + crown0
        C10 = ccol0 * n + (crown0+1)
        C11 = (ccol0+1) * n + (crown0+1)
        
        # M1 = (A00+A11)(B00+B11) -> C00:+1, C10:+1, C11:-1
        U[A00][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C10][col] = 1; W[C11][col] = -1
        col += 1
        
        # M2 = (A00+A11)(B10+B11) -> C10:+1, C11:+1
        U[A00][col] = 1; U[A11][col] = 1
        V[B10][col] = 1; V[B11][col] = 1
        W[C10][col] = 1; W[C11][col] = 1
        col += 1
        
        # M3 = A00(B00-B10) -> C01:+1, C11:+1
        U[A00][col] = 1
        V[B00][col] = 1; V[B10][col] = -1
        W[C01][col] = 1; W[C11][col] = 1
        col += 1
        
        # M4 = A11(B10+B11) -> C00:-1, C01:+1
        U[A11][col] = 1
        V[B10][col] = 1; V[B11][col] = 1
        W[C00][col] = -1; W[C01][col] = 1
        col += 1
        
        # M5 = (A01+A11)(B00+B10) -> C00:+1, C10:-1
        U[A01][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B10][col] = 1
        W[C00][col] = 1; W[C10][col] = -1
        col += 1
        
        # M6 = (A10-A00)(B01+B11) -> C11:+1
        U[A10][col] = 1; U[A00][col] = -1
        V[B01][col] = 1; V[B11][col] = 1
        W[C11][col] = 1
        col += 1
        
        # M7 = (A00-A10)(B10-B11) -> C00:+1, C10:-1
        U[A00][col] = 1; U[A10][col] = -1
        V[B10][col] = 1; V[B11][col] = -1
        W[C00][col] = 1; W[C10][col] = -1
        col += 1
    
    assert col == rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
