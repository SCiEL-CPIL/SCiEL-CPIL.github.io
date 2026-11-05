import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify the schoolbook decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        nm, r = len(U_mat), len(U_mat[0])
        mp = len(V_mat)
        np_ = len(W_mat)
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                s = 0.0
                                for a in range(r):
                                    s += U_mat[i*m+j][a] * V_mat[jp*p+k][a] * W_mat[kp*n+ip][a]
                                expected = (1.0 if (i==ip and j==jp and k==kp) else 0.0)
                                if abs(s - expected) > 1e-9:
                                    return False
        return True
    
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: return zeros (should not happen)
    return {"U": [[0]*rank_school for _ in range(n*m)], 
            "V": [[0]*rank_school for _ in range(m*p)], 
            "W": [[0]*rank_school for _ in range(p*n)]}
# EVOLVE-BLOCK-END
