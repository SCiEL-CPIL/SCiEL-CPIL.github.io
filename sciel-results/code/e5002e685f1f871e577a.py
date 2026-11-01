"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal exact tensor decomposition for 2x2 matrix multiplication.
    
    Uses Strassen's algorithm (rank 7) which is the known minimum for 2x2 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank = 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a, b], [c, d]] with indices: a=0, b=1, c=2, d=3
    # B = [[e, f], [g, h]] with indices: e=0, f=1, g=2, h=3
    # C = [[C00, C01], [C10, C11]] with indices: C00=0, C01=1, C10=2, C11=3
    
    # M1 = (a + d)(e + h)
    U[0][0] = 1
    U[3][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] = 1
    W[3][0] = 1
    
    # M2 = (c + d)e
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    W[3][1] = -1
    
    # M3 = a(f - h)
    U[0][2] = 1
    V[1][2] = 1
    V[3][2] = -1
    W[1][2] = 1
    W[3][2] = 1
    
    # M4 = d(g - e)
    U[3][3] = 1
    V[2][3] = 1
    V[0][3] = -1
    W[0][3] = 1
    W[2][3] = 1
    
    # M5 = (a + b)h
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    W[1][4] = 1
    
    # M6 = (c - a)(e + f)
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (b - d)(g + h)
    U[1][6] = 1
    U[3][6] = -1
    V[2][6] = 1
    V[3][6] = 1
    W[0][6] = 1
    
    # Verify the decomposition with random test cases
    import numpy as np
    np.random.seed(42)
    
    for _ in range(50):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute using decomposition
        U_mat = np.array(U)
        V_mat = np.array(V)
        W_mat = np.array(W)
        
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # Sum over c: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    total = 0
                    for c in range(rank):
                        total += U_mat[i*m + j, c] * V_mat[j*p + k, c] * W_mat[k*n + i, c]
                    C_computed[i, k] += total
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback: use schoolbook if verification fails
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
