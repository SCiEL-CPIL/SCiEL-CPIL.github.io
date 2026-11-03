"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen decomposition for a 2x2x2 block
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Local indices 0,1 map to global i0, i0+1 etc.
        I = [i0, i0 + 1]
        J = [j0, j0 + 1]
        K = [k0, k0 + 1]
        
        # M1: (A11+A22)(B11+B22) -> C11 + C22
        U[I[0]*m + J[0]][col] = 1
        U[I[1]*m + J[1]][col] = 1
        V[J[0]*p + K[0]][col] = 1
        V[J[1]*p + K[1]][col] = 1
        W[K[0]*n + I[0]][col] = 1
        W[K[1]*n + I[1]][col] = 1
        col += 1
        
        # M2: (A21+A22)B11 -> C21 - C22
        U[I[1]*m + J[0]][col] = 1
        U[I[1]*m + J[1]][col] = 1
        V[J[0]*p + K[0]][col] = 1
        W[K[0]*n + I[1]][col] = 1
        W[K[1]*n + I[1]][col] = -1
        col += 1
        
        # M3: A11(B12-B22) -> C12 + C22
        U[I[0]*m + J[0]][col] = 1
        V[J[0]*p + K[1]][col] = 1
        V[J[1]*p + K[1]][col] = -1
        W[K[1]*n + I[0]][col] = 1
        W[K[1]*n + I[1]][col] = 1
        col += 1
        
        # M4: A22(B21-B11) -> C21 + C11
        U[I[1]*m + J[1]][col] = 1
        V[J[1]*p + K[0]][col] = 1
        V[J[0]*p + K[0]][col] = -1
        W[K[0]*n + I[1]][col] = 1
        W[K[0]*n + I[0]][col] = 1
        col += 1
        
        # M5: (A11+A12)B22 -> -C12 - C11
        U[I[0]*m + J[0]][col] = 1
        U[I[0]*m + J[1]][col] = 1
        V[J[1]*p + K[1]][col] = 1
        W[K[1]*n + I[0]][col] = -1
        W[K[0]*n + I[0]][col] = -1
        col += 1
        
        # M6: (A21-A11)(B11+B12) -> C22
        U[I[1]*m + J[0]][col] = 1
        U[I[0]*m + J[0]][col] = -1
        V[J[0]*p + K[0]][col] = 1
        V[J[0]*p + K[1]][col] = 1
        W[K[1]*n + I[1]][col] = 1
        col += 1
        
        # M7: (A12-A22)(B21+B22) -> C11
        U[I[0]*m + J[1]][col] = 1
        U[I[1]*m + J[1]][col] = -1
        V[J[1]*p + K[0]][col] = 1
        V[J[1]*p + K[1]][col] = 1
        W[K[0]*n + I[0]][col] = 1
        col += 1

    # Add two Strassen blocks
    add_strassen(0, 0, 0) # j in {0,1}, k in {0,1}
    add_strassen(0, 0, 2) # j in {0,1}, k in {2,3}
    
    # Add schoolbook terms for the rest
    # j=2 (index 2), k in 0..4
    for i in range(n):
        for k in range(p):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # j in 0,1, k=4
    for j in range(2):
        for i in range(n):
            U[i*m + j][col] = 1
            V[j*p + 4][col] = 1
            W[4*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
