"""Program for matrix multiplication 2x3x3 tensor decomposition."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """Construct rank-minimal tensor decomposition for 2x3x3 matrix multiplication.
    
    Uses the schoolbook decomposition with rank = n*m*p = 18.
    The tensor T[(i,j), (j',k), (i',k')] = 1 iff i=i', j=j', k=k'.
    
    Factor matrices:
      U: (n*m) x rank = 6 x 18
      V: (m*p) x rank = 9 x 18  
      W: (p*n) x rank = 6 x 18
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 18
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
