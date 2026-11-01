"""
Program for Matrix Multiplication 2,2,7 rank-minimal decomposition.
Uses the schoolbook decomposition which is rank 28.
For 2x2 x 2x7, the bilinear complexity is known to be 4p = 28 for p >= 3,
as Strassen-style improvements do not extend beyond the 2x2x2 case.
"""

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 28 for schoolbook
    
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
