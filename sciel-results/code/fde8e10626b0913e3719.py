"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    # Helper to add Strassen block for j in {j0, j1}, k in {k0, k1}
    def add_strassen(j0, j1, k0, k1):
        nonlocal c
        # Factor 1
        U[0*m + j0][c] += 1
        U[1*m + j1][c] += 1
        V[j0*p + k0][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + 0][c] += 1
        W[k1*n + 1][c] += 1
        c += 1
        
        # Factor 2
        U[1*m + j0][c] += 1
        U[1*m + j1][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + 1][c] += 1
        W[k1*n + 1][c] += -1
        c += 1
        
        # Factor 3
        U[0*m + j0][c] += 1
        V[j1*p + k0][c] += 1
        V[j1*p + k1][c] += -1
        W[k1*n + 0][c] += 1
        W[k1*n + 1][c] += 1
        c += 1
        
        # Factor 4
        U[1*m + j1][c] += 1
        V[j0*p + k1][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + 0][c] += 1
        W[k0*n + 1][c] += 1
        c += 1
        
        # Factor 5
        U[0*m + j0][c] += 1
        U[0*m + j1][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + 0][c] += -1
        W[k1*n + 0][c] += 1
        c += 1
        
        # Factor 6
        U[1*m + j0][c] += 1
        U[0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + k1][c] += 1
        W[k1*n + 1][c] += 1
        c += 1
        
        # Factor 7
        U[0*m + j1][c] += 1
        U[1*m + j1][c] += -1
        V[j1*p + k0][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + 0][c] += 1
        W[k1*n + 0][c] += 1
        c += 1

    # Block 1: j in {0,1}, k in {0,1}
    add_strassen(0, 1, 0, 1)
    
    # Block 2: j in {0,1}, k in {2,3}
    add_strassen(0, 1, 2, 3)
    
    # Block 3: j in {0,1}, k in {4} -> Schoolbook 2x2x1
    for i in range(n):
        for j in range(2):
            U[i*m + j][c] += 1
            V[j*p + 4][c] += 1
            W[4*n + i][c] += 1
            c += 1
            
    # Block 4: j in {2}, k in {0..4} -> Schoolbook 2x1x5
    for i in range(n):
        for k in range(p):
            U[i*m + 2][c] += 1
            V[2*p + k][c] += 1
            W[k*n + i][c] += 1
            c += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
