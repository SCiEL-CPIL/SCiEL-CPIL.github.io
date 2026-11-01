"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    A, B, C = n*m, m*p, p*n  # 8, 32, 16
    
    # Build the tensor as a C x (A*B) matrix to check rank lower bound
    # T[a,b,c] = 1 iff b == (a - (c%n)*m)*p + c//n
    M = np.zeros((C, A*B), dtype=float)
    for c in range(C):
        i = c % n
        k = c // n
        for a in range(A):
            j = a - i * m
            if 0 <= j < m:
                b = j * p + k
                if 0 <= b < B:
                    M[c, a*B + b] = 1
    
    # Rank of this matrix is a lower bound on tensor rank
    rank_lower = np.linalg.matrix_rank(M)
    
    # Schoolbook rank
    rank_school = n * m * p
    
    # If lower bound equals schoolbook, schoolbook is optimal
    if rank_lower >= rank_school:
        rank = rank_school
    else:
        # Try to find a better decomposition (for now, fall back to schoolbook)
        rank = rank_school
    
    U = [[0] * rank for _ in range(A)]
    V = [[0] * rank for _ in range(B)]
    W = [[0] * rank for _ in range(C)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] =1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
