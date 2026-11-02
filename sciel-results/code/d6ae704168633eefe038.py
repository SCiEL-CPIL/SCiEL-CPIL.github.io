"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication.
Uses schoolbook rank 50 as a safe baseline."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = len(U[0])
        # Check diagonal: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        # Check off-diagonal: for all (i,j,k) != (i3,j2,k3), sum should be 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i3 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i == i3 and j == j2 and k == k2:
                                    continue
                                s = 0.0
                                for c in range(r):
                                    s += U[i*m+j][c] * V[j2*p+k2][c] * W[k*n+i3][c]
                                if abs(s) > tol:
                                    return False
        return True
    
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    # Directly return schoolbook rank 50 decomposition
    # Rank-25 attempt is invalid for n>1 due to lack of i-dependence in W
    U, V, W = schoolbook(n, m, p)
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Last resort: return schoolbook even if verification failed
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
