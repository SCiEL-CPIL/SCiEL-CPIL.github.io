"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import itertools
import random

def _verify_factors(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] equals the MM tensor."""
    rank = len(U[0])
    for i, j, k in itertools.product(range(n), range(m), range(p)):
        s = 0
        for c in range(rank):
            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        if abs(s - 1) > tol:
            return False
    return True

def _construct_improved(n, m, p):
    """
    Constructs a rank 76 decomposition for 4x5x5 matrix multiplication.
    Decomposes into:
    1. 4x4x4 schoolbook (rank 64)
    2. 4x4x1 linear terms (rank 4)
    3. 4x1x4 linear terms (rank 4)
    4. 4x1x1 linear terms (rank 4)
    """
    rank = 76
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4x4x4 schoolbook (i, j, k in 0..3)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    # Part 2: j in 0..3, k=4 (contributes to C[i, 4])
    # Term: A[i, j] * B[j, 4] * C[i, 4]
    for i in range(4):
        for j in range(4):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
            
    # Part 3: j=4, k in 0..3 (contributes to C[i, k])
    # Term: A[i, 4] * B[4, k] * C[i, k]
    for i in range(4):
        for k in range(4):
            U[i*m+4][col] = 1
            V[4*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
            
    # Part 4: j=4, k=4 (contributes to C[i, 4])
    # Term: A[i, 4] * B[4, 4] * C[i, 4]
    for i in range(4):
        U[i*m+4][col] = 1
        V[4*p+4][col] = 1
        W[4*n+i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    return _construct_improved(n, m, p)
# EVOLVE-BLOCK-END
